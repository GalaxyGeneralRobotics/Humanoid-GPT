"""Main entry point for tracking deployment (simulation and real robot).

Usage:
    # Simulation (offline tracking)
    python -m deploy.play_track --track_dir storage/data/exps

    # Simulation (online retarget, Noitom PNLink)
    python -m deploy.play_track --track_dir storage/data/exps --mocap_type pnlink

    # Simulation (online retarget, Xsens MVN over TCP on port 9763)
    python -m deploy.play_track --track_dir storage/data/exps \
        --mocap_type xsens --xsens_protocol tcp --xsens_port 9763

    # Real robot
    python -m deploy.play_track --real --net enx6c1ff7579fdf

Modes (keyboard number keys):
    0 = Walk policy
    1 = Online retarget (mocap)
    2+ = Offline tracking (reference trajectories from track_dir)
"""

from __future__ import annotations

import os
import time
import signal
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass

import tyro
import numpy as np
from jax import tree_util as jtu
from loop_rate_limiters import RateLimiter

from tracking import constants as consts
from tracking.constants import KPT_NAMES
from tracking.convert_qpos2kpt import qpos2kpt
from utils.transforms_np import base2navi, quat2mat
from utils.sim_mj import get_sensor_data as mj_sensor
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from tracking.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config, apply_ema_qpos

from deploy.walk_policy import WalkPolicy
from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.constants import DEFAULT_QPOS as DEFAULT_QPOS_JOINT, KPs_walking, KDs_walking
from tracking.metrics import (
    calculate_kpt_mae_error,
    calculate_joint_tracking_error,
    calculate_joint_jerk,
    calculate_root_tracking_error,
    calculate_trajectory_length,
    calculate_max_errors,
)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402  – needed for DeployKeyboardCMD fork workaround


# ---------------------------------------------------------------------------
# Mocap buffer with caching and linear extrapolation
# ---------------------------------------------------------------------------

class MocapBuffer:
    """Read the latest mocap data from the shared buffer."""

    def __init__(self, buf, ts):
        self._buf = buf
        self._ts = ts

    def read(self) -> tuple[np.ndarray, float]:
        from deploy.retarget import read_mocap_buffer
        qpos_full, ts = read_mocap_buffer(self._buf, self._ts)
        return qpos_full, ts


# ---------------------------------------------------------------------------
# Live reference converter: qpos_full -> ref_state dict for G1TrackInferFn
# ---------------------------------------------------------------------------

def _batch_rot_log_so3(R: np.ndarray) -> np.ndarray:
    """Batch SO(3) log for K rotation matrices. R: (K,3,3) -> (K,3)."""
    tr = np.trace(R, axis1=1, axis2=2)  # (K,)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (K,)
    skew = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], axis=1)  # (K, 3)
    small = theta < 1e-6
    safe_theta = np.where(small, 1.0, theta)
    k = np.where(small, 0.5, theta / (2.0 * np.sin(safe_theta)))
    return k[:, None] * skew


def _batch_pose_delta_to_twist(T_prev: np.ndarray, T_curr: np.ndarray,
                               dt: float) -> np.ndarray:
    """Batch compute 6D twists (ang, lin) in world. T: (K,4,4) -> (K,6)."""
    R0, p0 = T_prev[:, :3, :3], T_prev[:, :3, 3]
    R1, p1 = T_curr[:, :3, :3], T_curr[:, :3, 3]
    v_w = (p1 - p0) / dt
    R_delta = np.einsum("kij,kjl->kil", R0.transpose(0, 2, 1), R1)
    rotvec_body = _batch_rot_log_so3(R_delta)
    w_w = np.einsum("kij,kj->ki", R0, rotvec_body / dt)
    return np.concatenate([w_w, v_w], axis=1).astype(np.float32)


def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (z-rotation) from a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LiveRefConverter:
    """Convert a raw qpos_full (from mocap) to the ref_state dict expected
    by G1TrackInferFn, computing keypoint poses via MuJoCo FK.

    Keypoint velocities are computed via finite differences between
    consecutive frames, matching the pre-computed kpt_cvel_in_gv used
    in training reference data.

    For real-robot deployment, call ``set_robot_initial_pose()`` when
    entering tracking mode so that the reference root pose is rebiased
    to align with the robot's initial frame (IMU yaw + phantom [0,0,z]).
    """

    def __init__(self, mj_model: mujoco.MjModel, ctrl_dt: float):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.ctrl_dt = ctrl_dt
        self.kpt_body_ids = np.array([mj_model.body(n).id for n in KPT_NAMES])
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        # Initial-pose calibration (P3): set via set_robot_initial_pose()
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def reset(self):
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def set_robot_initial_pose(self, robot_quat: np.ndarray, robot_xy: np.ndarray):
        """Record the robot's pose at the moment tracking begins.
        Called once per tracking session so that reference poses can be
        rebiased into the robot's coordinate frame."""
        self._robot_init_yaw = _quat_to_yaw(robot_quat)
        self._robot_init_xy = robot_xy[:2].copy()

    def _rebias_qpos(self, qpos_full: np.ndarray) -> np.ndarray:
        """Rebias reference root (xy, yaw) so that the first frame aligns
        with the robot's initial pose recorded by set_robot_initial_pose().
        This makes update_coord_cmd() compute correct relative differences
        on the real robot where the phantom state starts at [0,0,0.78]."""
        if self._robot_init_yaw is None:
            return qpos_full

        qpos = qpos_full.copy()

        # Capture reference initial pose on first call
        if self._ref_init_xy is None:
            self._ref_init_yaw = _quat_to_yaw(qpos[3:7])
            self._ref_init_xy = qpos[:2].copy()

        # Compute delta from reference initial
        ref_yaw = _quat_to_yaw(qpos[3:7])
        d_yaw = _wrap_to_pi(ref_yaw - self._ref_init_yaw)
        d_xy = qpos[:2] - self._ref_init_xy

        # Rotate d_xy by the yaw offset between robot and reference initial
        yaw_offset = _wrap_to_pi(self._robot_init_yaw - self._ref_init_yaw)
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        rotated_dxy = np.array([c * d_xy[0] - s * d_xy[1],
                                s * d_xy[0] + c * d_xy[1]], dtype=np.float32)

        # Apply to robot initial pose
        qpos[:2] = self._robot_init_xy + rotated_dxy
        new_yaw = self._robot_init_yaw + d_yaw

        # Apply yaw correction in WORLD frame: q_new = q_dz * q_ref.
        # Right-multiplication applies a BODY-frame yaw and distorts roll/pitch
        # when the torso is not upright (e.g. bending/squatting).
        d_yaw_apply = _wrap_to_pi(new_yaw - ref_yaw)
        c2, s2 = np.cos(d_yaw_apply / 2), np.sin(d_yaw_apply / 2)
        q_dz = np.array([c2, 0.0, 0.0, s2], dtype=np.float32)
        q_new = _quat_mul_wxyz(q_dz, qpos[3:7])
        qpos[3:7] = q_new / np.clip(np.linalg.norm(q_new), 1e-8, None)

        return qpos

    def convert(self, qpos_full: np.ndarray) -> dict:
        """Convert a single qpos_full (36,) to ref_state dict (batch dim = 1)."""
        qpos_full = self._rebias_qpos(qpos_full)
        self.mj_data.qpos[:] = qpos_full
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Gravity-view frame
        base2wrd_rot = quat2mat(qpos_full[3:7])
        gvi2wrd_rot = base2navi(base2wrd_rot)
        gvi2wrd_pose = np.eye(4, dtype=np.float32)
        gvi2wrd_pose[:2, 3] = qpos_full[:2]
        gvi2wrd_pose[:3, :3] = gvi2wrd_rot

        # Keypoint poses in world (vectorised)
        num_kpt = len(self.kpt_body_ids)
        kpt2wrd_pose = np.tile(np.eye(4, dtype=np.float32), (num_kpt, 1, 1))
        kpt2wrd_pose[:, :3, 3] = self.mj_data.xpos[self.kpt_body_ids]
        kpt2wrd_pose[:, :3, :3] = self.mj_data.xmat[self.kpt_body_ids].reshape(-1, 3, 3)

        # Transform to gravity-view
        kpt2gv_pose = np.linalg.inv(gvi2wrd_pose) @ kpt2wrd_pose  # (K,4,4)

        # Keypoint velocities via finite differences (vectorised, no scipy)
        kpt_cvel_in_wrd = np.zeros((num_kpt, 6), dtype=np.float32)
        if self._prev_kpt2wrd_pose is not None:
            kpt_cvel_in_wrd = _batch_pose_delta_to_twist(
                self._prev_kpt2wrd_pose, kpt2wrd_pose, self.ctrl_dt
            )
        self._prev_kpt2wrd_pose = kpt2wrd_pose.copy()

        # Rotate world-frame velocities to gravity-view frame
        R_wrd2gv = gvi2wrd_pose[:3, :3].T
        kpt_cvel_in_gv = np.zeros_like(kpt_cvel_in_wrd)
        kpt_cvel_in_gv[:, :3] = kpt_cvel_in_wrd[:, :3] @ R_wrd2gv.T
        kpt_cvel_in_gv[:, 3:] = kpt_cvel_in_wrd[:, 3:] @ R_wrd2gv.T

        # Gravity-view velocity (root frame linear + yaw angular)
        gv_vel = np.zeros(3, dtype=np.float32)
        if self._prev_gv2wrd_pose is not None:
            curr2prev = np.linalg.inv(self._prev_gv2wrd_pose) @ gvi2wrd_pose
            gv_vel[:2] = curr2prev[:2, 3] / self.ctrl_dt
            gv_vel[2] = np.arctan2(curr2prev[1, 0], curr2prev[0, 0]) / self.ctrl_dt
        self._prev_gv2wrd_pose = gvi2wrd_pose.copy()

        # Joint velocity from finite differences
        qvel = np.zeros(35, dtype=np.float32)
        if self._prev_qpos is not None:
            qvel[6:] = (qpos_full[7:] - self._prev_qpos[7:]) / self.ctrl_dt
        self._prev_qpos = qpos_full.copy()

        # Build ref_state dict with batch dimension
        return {
            "qpos": qpos_full[None].astype(np.float32),
            "qvel": qvel[None].astype(np.float32),
            "kpt2gv_pose": kpt2gv_pose[None].astype(np.float32),
            "kpt_cvel_in_gv": kpt_cvel_in_gv[None].astype(np.float32),
            "gv2wrd_pose": gvi2wrd_pose[None].astype(np.float32),
            "gv_vel": gv_vel[None].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Reference motion loading
# ---------------------------------------------------------------------------

def load_offline_motions(track_dir: str, mj_model: mujoco.MjModel, freq: int = 50) -> list[dict]:
    """Load .npz reference trajectories, apply EMA and convert to kpt format.

    Returns list of dicts. Each dict has numpy-array fields (safe for
    jtu.tree_map) plus a ``_filename`` key that is excluded before tree_map.
    """
    folder = Path(track_dir)
    if folder.is_file():
        files = [folder]
    else:
        files = sorted(folder.glob("*.npz"))

    motions = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            print(f"[WARN] Skipping {f.name}: no qpos field")
            continue

        data["qpos"] = apply_ema_qpos(data["qpos"])
        freq_src = float(data.get("frequency", 50))
        kpt_data = qpos2kpt(
            mj_model, np.float32(data["qpos"]),
            freq_src=freq_src, freq_tgt=freq,
            interp_sec=0.5, end_default_sec=0.5,
            debug=False, foot_contact_est=False,
            height_clip_mode=None, video_path=None,
        )
        motions.append({"data": kpt_data, "filename": f.name})
        print(f"  Mode {len(motions)+1}: {f.name} ({len(kpt_data['qpos'])} frames)")

    return motions


# ---------------------------------------------------------------------------
# Offline tracking metrics (same format as inference.py)
# ---------------------------------------------------------------------------

def _print_offline_metrics(traj_metrics: dict, filename: str, ref_traj: dict, mj_model, ctrl_dt: float = 0.02) -> None:
    """Print tracking error metrics in the same format as scripts/inference.py."""
    actual_traj_len = len(ref_traj["qpos"])
    traj_length_ratio, termination_step = calculate_trajectory_length(
        traj_metrics["state_history"], ref_traj, mj_model
    )
    avg_kpt_pos = np.mean(traj_metrics["kpt_pos_errors"])
    avg_kpt_rot = np.mean(traj_metrics["kpt_rot_errors"])
    avg_joint_pos = np.mean(traj_metrics["joint_pos_errors"])
    avg_joint_vel = np.mean(traj_metrics["joint_vel_errors"])
    avg_root_pos = np.mean(traj_metrics["root_pos_errors"])
    avg_root_vel = np.mean(traj_metrics["root_vel_errors"])
    avg_root_yaw = np.mean(traj_metrics["root_yaw_errors"])
    avg_joint_jerk = calculate_joint_jerk(traj_metrics["state_history"], ctrl_dt)
    max_errors = calculate_max_errors(traj_metrics)

    print(f"\n  [Offline Track] {filename} completed:")
    print(f"    Completion: {traj_length_ratio:.4f} ({termination_step}/{actual_traj_len} steps)")
    print(f"    KPT Position MAE: {avg_kpt_pos:.6f} m (Max: {max_errors['max_kpt_pos_error']:.6f} m)")
    print(f"    KPT Rotation MAE: {avg_kpt_rot:.6f} rad (Max: {max_errors['max_kpt_rot_error']:.6f} rad)")
    print(f"    Joint Position MAE: {avg_joint_pos:.6f} rad (Max: {max_errors['max_joint_pos_error']:.6f} rad)")
    print(f"    Joint Velocity MAE: {avg_joint_vel:.6f} rad/s (Max: {max_errors['max_joint_vel_error']:.6f} rad/s)")
    print(f"    Joint Jerk: {avg_joint_jerk:.3f} rad/s^3")
    print(f"    Root Pos Error: {avg_root_pos:.3f} mm (Max: {max_errors['max_root_pos_error']:.3f} mm)")
    print(f"    Root Vel Error: {avg_root_vel:.3f} mm/s (Max: {max_errors['max_root_vel_error']:.3f} mm/s)")
    print(f"    Root Yaw Error: {avg_root_yaw:.6f} rad (Max: {max_errors['max_root_yaw_error']:.6f} rad)\n")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_sim(args: "DeployArgs"):
    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    # Load ONNX tracking policy
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
    )
    track_policy = get_policy_onnx(policy_args)

    # Load walk policy
    walk_policy = WalkPolicy(args.onnx_walk)

    # Load offline reference motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    # Keyboard
    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    # MuJoCo sim (correct sim_dt=0.001 from Humanoid-GPT)
    init_qpos = consts.DEFAULT_QPOS.copy()
    mj_sim = G1TrackMjSim(init_qpos=init_qpos, headless=False, ctrl_dt=ctrl_dt)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, track_policy, privileged=False)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    # Camera view: match inference.py
    if not mj_sim.headless and mj_sim.viewer is not None:
        viewer_cam = mj_sim.viewer.cam
        viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_cam.trackbodyid = 0
        viewer_cam.azimuth = 90.0
        viewer_cam.elevation = -20.0
        viewer_cam.distance = 2.0

    # Live retarget converter
    live_converter = LiveRefConverter(mj_sim.mj_model, ctrl_dt)

    # Online retarget (optional)
    mocap_buffer = None
    buf_hand = None
    if not args.no_mocap:
        try:
            from deploy.retarget import start_realtime_retarget
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=args.mocap_type,
                buffer_ms=args.buffer_ms,
                xsens_host=args.xsens_host,
                xsens_port=args.xsens_port,
                xsens_protocol=args.xsens_protocol,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")

    rate = RateLimiter(frequency=freq, warn=False)
    last_mode = 0
    track_step = 0
    ref_traj = None
    traj_metrics = None  # For offline tracking: kpt/joint/root errors, state_history
    traj_filename = None
    prev_online_ref = None

    print("\n=== Simulation ready. Press number keys to switch modes. ===\n")

    try:
        while True:
            if keyboard.check_reset_request():
                state = mj_sim.reset(state)
                infer_fn.info["last_action"][:] = 0
                infer_fn.info["step"] = 0
                live_converter.reset()
                print("[Reset] Simulation reset.")

            cmd = keyboard.step_command()
            mode = cmd.mode

            # Mode transitions
            entering_track = (last_mode == 0) and (mode >= 1)
            leaving_track = (last_mode >= 1) and (mode == 0)

            if entering_track:
                infer_fn.info["last_action"][:] = 0
                live_converter.reset()
                prev_online_ref = None
                if mode >= 2:
                    traj_idx = mode - 2
                    if traj_idx < len(ref_motions):
                        ref_traj = ref_motions[traj_idx]["data"]
                        track_step = 0
                        traj_filename = ref_motions[traj_idx]["filename"]
                        traj_metrics = {
                            "kpt_pos_errors": [],
                            "kpt_rot_errors": [],
                            "joint_pos_errors": [],
                            "joint_vel_errors": [],
                            "root_pos_errors": [],
                            "root_vel_errors": [],
                            "root_yaw_errors": [],
                            "state_history": [],
                        }
                        print(f"[Track] Start offline: {traj_filename}")
                    else:
                        print(f"[Track] Invalid trajectory index {traj_idx}")
                        mode = 0
                elif mode == 1:
                    print("[Track] Start online retarget")

            if leaving_track:
                if last_mode >= 2 and traj_metrics is not None and len(traj_metrics["state_history"]) > 0:
                    _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model, ctrl_dt)
                traj_metrics = None
                traj_filename = None
                live_converter.reset()
                print("[Track] Back to walk mode")

            if mode == 0:
                # Walk policy
                cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
                gyro = mj_sensor(mj_sim.mj_model, state.mj_data, "gyro_pelvis")
                motor_targets = walk_policy.infer(
                    state.mj_data.qpos[3:7], gyro,
                    state.mj_data.qpos[7:], state.mj_data.qvel[6:],
                    cmd_vel,
                )
                # Walk uses different PD gains than tracking
                for _ in range(mj_sim.num_sim_substeps):
                    torques = KPs_walking * (motor_targets - state.mj_data.qpos[7:]) + KDs_walking * (-state.mj_data.qvel[6:])
                    torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
                    state.mj_data.ctrl[:] = torques
                    mujoco.mj_step(mj_sim.mj_model, state.mj_data)

            elif mode == 1:
                # Online retarget
                if mocap_buffer is not None:
                    qpos_full, _ = mocap_buffer.read()
                    ref_new = live_converter.convert(qpos_full)
                    if prev_online_ref is None:
                        ref_curr = ref_new
                    else:
                        ref_curr = prev_online_ref
                    ref_next = ref_new
                    prev_online_ref = ref_new
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

            else:
                # Offline tracking (mode >= 2)
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

                    # Collect metrics (same as inference.py)
                    if traj_metrics is not None:
                        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(
                            state, ref_curr, ref_next, mj_sim.mj_model
                        )
                        joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(
                            state, ref_curr
                        )
                        root_pos_err_mm, root_vel_err_mms, root_yaw_err = calculate_root_tracking_error(
                            state, ref_curr
                        )
                        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
                        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
                        traj_metrics["joint_pos_errors"].append(joint_pos_mae)
                        traj_metrics["joint_vel_errors"].append(joint_vel_mae)
                        traj_metrics["root_pos_errors"].append(root_pos_err_mm)
                        traj_metrics["root_vel_errors"].append(root_vel_err_mms)
                        traj_metrics["root_yaw_errors"].append(root_yaw_err)
                        traj_metrics["state_history"].append({
                            "qpos": state.mj_data.qpos.copy(),
                            "qvel": state.mj_data.qvel.copy(),
                            "xpos": state.mj_data.xpos.copy(),
                            "xmat": state.mj_data.xmat.copy(),
                        })

                    track_step = track_step + 1

                    # Print metrics when trajectory completes (after processing last frame)
                    if track_step >= traj_len and traj_metrics is not None and len(traj_metrics["state_history"]) > 0:
                        _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model, ctrl_dt)
                        traj_metrics = None
                        traj_filename = None
                        ref_traj = None  # Avoid index out of bounds on next iteration

            last_mode = mode
            mj_sim.view(state)
            rate.sleep()

            if cmd.kill:
                break

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.close()
        print("[Sim] Exited.")


# ---------------------------------------------------------------------------
# Real-robot loop
# ---------------------------------------------------------------------------

def run_real(args: "DeployArgs"):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.utils.thread import RecurrentThread
    from deploy.real_robot import LowLevelControlG1, KeyMap
    from deploy.hand_control import make_hand_controller, update_hand_from_mocap
    from deploy.retarget import (
        start_realtime_retarget,
        read_hand_buffer,
    )

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt = ctrl_dt)

    # Track policy (TensorRT for real-robot inference)
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
    )
    track_policy = get_policy_onnx(policy_args, use_trt=True, strict_trt=True)
    walk_policy = WalkPolicy(args.onnx_walk)

    # Offline motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    # Pre-init pygame then quit video so the fork in KeyboardCmdPad
    # does not inherit an X11 fd (avoids "X connection broken").
    pygame.init()
    pygame.display.quit()

    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    # Robot + phantom sim for reference FK only (no robot-side FK needed)
    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)

    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = consts.load_mj_model(xml_path)
    phantom_model.opt.timestep = 0.001

    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # Online retarget (optional)
    mocap_buffer = None
    buf_hand = None
    if not args.no_mocap:
        try:
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                robot="unitree_g1", dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=args.mocap_type,
                buffer_ms=args.buffer_ms,
                xsens_host=args.xsens_host,
                xsens_port=args.xsens_port,
                xsens_protocol=args.xsens_protocol,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")
    else:
        print("[Mocap] Disabled (--no-mocap). Online mode disabled.")

    # This script only drives the Dex3 hand (binary open/close from mocap via
    # update_hand_from_mocap); other mounted hands are skipped with a hint.
    hand_ctrl = make_hand_controller(net=args.net, supported=("dex3",)) if args.enable_hand else None

    last_mode = 0
    track_step = 0
    ref_traj = None
    last_left_hand = None
    last_right_hand = None
    prev_online_ref = None

    def locomotion_step():
        nonlocal last_mode, track_step, ref_traj, last_left_hand, last_right_hand, prev_online_ref

        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        cmd = keyboard.step_command()
        mode = cmd.mode

        entering_track = (last_mode == 0) and (mode >= 1)
        leaving_track = (last_mode >= 1) and (mode == 0)

        if entering_track:
            infer_fn.info["last_action"][:] = 0
            live_converter.reset()
            prev_online_ref = None
            # Calibrate reference frame to robot's current pose
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            live_converter.set_robot_initial_pose(root_quat, robot_xy)
            if mode >= 2:
                traj_idx = mode - 2
                if traj_idx < len(ref_motions):
                    ref_traj = ref_motions[traj_idx]["data"]
                    track_step = 0
            elif mode == 1 and mocap_buffer is None:
                print("[Track] Online retarget unavailable (--no-mocap or init failed)")

        if leaving_track:
            live_converter.reset()

        if mode == 0:
            cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
            motor_targets = walk_policy.infer(root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel)
            low_ctrl.step(motor_targets, KPs_walking, KDs_walking)
        else:
            if mode == 1:
                if mocap_buffer is None:
                    last_mode = mode
                    return
                qpos_full, _ = mocap_buffer.read()
                ref_new = live_converter.convert(qpos_full)
                if prev_online_ref is None:
                    ref_curr = ref_new
                else:
                    ref_curr = prev_online_ref
                ref_next = ref_new
                prev_online_ref = ref_new
            else:
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    track_step = min(track_step + 1, traj_len - 1)
                else:
                    last_mode = mode
                    return

            motor_targets = infer_fn.infer_onnx_real(
                root_quat, root_gyro, jnt_qpos, jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs)

            if hand_ctrl is not None and buf_hand is not None:
                hand_cmd = read_hand_buffer(buf_hand)
                last_left_hand, last_right_hand = update_hand_from_mocap(
                    hand_ctrl, hand_cmd, last_left_hand, last_right_hand,
                )

        last_mode = mode

    # Startup sequence
    print("<Mode: Damping> Waiting for <start> on remote...")
    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        time.sleep(ctrl_dt)

    low_ctrl.move_to_default_pos(duration=2.0)
    print("<Mode: Default> Waiting for <A> on remote...")
    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        time.sleep(ctrl_dt)

    print("<Mode: Locomotion> Starting control loop...")
    loco_thread = RecurrentThread(interval=ctrl_dt, target=locomotion_step, name="loco")
    loco_thread.Start()

    running = True

    def _sigint(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    try:
        while running:
            if low_ctrl.remote.button[KeyMap.select] == 1:
                print("[Real] Select pressed, stopping.")
                running = False
            time.sleep(0.05)
    finally:
        low_ctrl.set_motor_damping()
        keyboard.close()
        print("[Real] Exited.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class DeployArgs:
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    track_dir: str = "storage/test"
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    convert_xml_path: str = str(consts.TRACK_XML)
    real: bool = False
    debug: bool = False
    freq: int = 50

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"  # one of: pnlink | xsens
    human_height: float = 1.7
    visualize_retarget: bool = True
    buffer_ms: float = 30.0

    # Xsens MVN streamer (only used when --mocap_type xsens).
    xsens_host: str = "0.0.0.0"      # local bind address for the receiver
    xsens_port: int = 9763           # default MVN Network Streamer port
    xsens_protocol: str = "tcp"      # "tcp" or "udp" - match MVN Studio setting

    # Real robot
    net: str = "enx6c1ff76e8ef5"
    enable_hand: bool = False


def main(args: DeployArgs):
    if args.real:
        run_real(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main(tyro.cli(DeployArgs))
