"""Tracking deployment with BrainCo hand control.

This module intentionally lives next to, but does not modify,
``deploy.play_track``.  The locomotion/tracking logic is kept aligned with the
original tracker, while the real-robot path adds a BrainCo hand qpos stream for
Noitom/PNLink mocap and sends it through
``deploy.brainco.brainco_controller.BraincoController``.

Usage:
    # Simulation: delegates to the original tracker.
    python -m deploy.play_track_nao --track_dir storage/data/exps

    # Real robot with BrainCo hands.
    python -m deploy.play_track_nao --real --net enx6c1ff7579fdf
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import tyro
import pygame
import mujoco
import numpy as np
from jax import tree_util as jtu

from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.play_track import DeployArgs as PlayTrackArgs
from deploy.play_track import LiveRefConverter, MocapBuffer, run_sim
from deploy.constants import DEFAULT_QPOS as DEFAULT_QPOS_JOINT, KDs_walking, KPs_walking

from deploy.retarget import _visualize_worker
from deploy.walk_policy import WalkPolicy
from tracking.policy import Args as PolicyArgs, get_policy_onnx

from tracking import constants as consts
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import G1TrackInferFn, apply_ema_qpos, g1_infer_env_config


# ---------------------------------------------------------------------------
# BrainCo hand conversion
# ---------------------------------------------------------------------------

_BRAINCO_TARGETS = ("brainco", "brainco2", "brainco3")
_HAND_CMD_DOF = 12


def _brainco_hand12_to_ctrl6(hand_qpos: np.ndarray) -> np.ndarray:
    """Convert one 12-D BrainCo hand qpos to 6 actuator commands.

    ``deploy.bvh_noitom_streamer`` emits each hand as:
        thumb(4), index(2), middle(2), ring(2), pinky(2)

    ``BraincoController`` expects:
        thumb, thumb_aux, index, middle, ring, pinky

    The scale factors are the same mapping previously used in our offline
    replay script.
    """
    q = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    if q.size < 12:
        return np.zeros(6, dtype=np.float32)

    ctrl = np.zeros(6, dtype=np.float32)
    ctrl[1] = q[0] / 1.5184
    ctrl[0] = (q[1] + q[2]) / (1.0472 + 1.0472)
    ctrl[2] = (q[4] + q[5]) / (1.4661 + 1.6930)
    ctrl[3] = (q[6] + q[7]) / (1.4661 + 1.6930)
    ctrl[4] = (q[8] + q[9]) / (1.4661 + 1.6930)
    ctrl[5] = (q[10] + q[11]) / (1.4661 + 1.6930)
    return np.clip(ctrl, 0.0, 1.0)


def brainco_qpos24_to_cmd12(
    hand_qpos: np.ndarray | None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert 24-D retargeted qpos to BrainCoController's 12-D qpos command.

    Retarget qpos order is left hand then right hand.  The controller command
    order is right hand then left hand.
    """
    if hand_qpos is None:
        return np.zeros(_HAND_CMD_DOF, dtype=np.float32)

    q = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    if q.size < 24:
        return np.zeros(_HAND_CMD_DOF, dtype=np.float32)

    cmd = np.zeros(_HAND_CMD_DOF, dtype=np.float32)
    left_qpos = q[:12]
    right_qpos = q[12:24]
    cmd[:6] = _brainco_hand12_to_ctrl6(right_qpos)
    cmd[6:] = _brainco_hand12_to_ctrl6(left_qpos)
    return np.clip(cmd * float(scale), 0.0, 1.0)


class BraincoHandSmoother:
    def __init__(self, alpha: float, scale: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.scale = scale
        self.last_cmd: np.ndarray | None = None

    def reset(self, value: np.ndarray | None = None) -> np.ndarray:
        self.last_cmd = (
            np.zeros(_HAND_CMD_DOF, dtype=np.float32)
            if value is None
            else np.asarray(value, dtype=np.float32).reshape(_HAND_CMD_DOF).copy()
        )
        return self.last_cmd.copy()

    def update_from_qpos24(self, hand_qpos: np.ndarray | None) -> np.ndarray:
        target = brainco_qpos24_to_cmd12(hand_qpos, scale=self.scale)
        if self.last_cmd is None or self.alpha >= 1.0:
            self.last_cmd = target
        else:
            self.last_cmd = self.last_cmd * (1.0 - self.alpha) + target * self.alpha
        return self.last_cmd.copy()


# ---------------------------------------------------------------------------
# Realtime retarget with BrainCo hand qpos
# ---------------------------------------------------------------------------

_HAND_JOINTS = {
    "left": {
        "wrist": "LeftHand",
        "tips": ["LeftHandIndex3", "LeftHandMiddle3", "LeftHandRing3", "LeftHandPinky3"],
    },
    "right": {
        "wrist": "RightHand",
        "tips": ["RightHandIndex3", "RightHandMiddle3", "RightHandRing3", "RightHandPinky3"],
    },
}
_RETARGET_SESSIONS: list[dict] = []


def _detect_hand_open(frame, wrist: str, tips: list[str], threshold: float = 0.05):
    try:
        if wrist == "RightHand":
            dist = np.linalg.norm(
                np.array(frame["RightHandIndex3"][0]) - np.array(frame["RightHandThumb3"][0])
            )
        else:
            dist = np.linalg.norm(
                np.array(frame["LeftHandIndex3"][0]) - np.array(frame["LeftHandThumb3"][0])
            )
        return dist > threshold, dist
    except KeyError:
        return False, 0.0


def _retarget_worker_with_brainco_hands(
    buf,
    buf_hand,
    buf_hand_qpos,
    ts,
    ready_evt,
    stop_evt,
    robot,
    actual_human_height,
    mocap_type,
    buffer_ms,
    hand_target,
    rt_pin,
):
    if rt_pin is not None:
        import os as _os

        cpu_id, fifo_prio = rt_pin
        try:
            _os.sched_setaffinity(0, {int(cpu_id)})
            _os.sched_setscheduler(0, _os.SCHED_FIFO, _os.sched_param(int(fifo_prio)))
        except (OSError, PermissionError):
            pass

    from deploy.brainco.noitom_hand_retarget import _retarget_noitom_hand_qpos
    from gmr import GeneralMotionRetargeting as GMR

    if (mocap_type or "").lower() == "pnlink":
        from noitom import NoitomClient

        client = NoitomClient()
        client.start_thread()
        get_frame = lambda: client.get_frame_data(timeout=True)
        src_human = "fbx_noitom"
    else:
        raise ValueError(f"Unknown mocap_type: {mocap_type}")

    retarget = GMR(src_human=src_human, tgt_robot=robot, actual_human_height=actual_human_height)
    qpos_last = None
    ema_alpha = 0.75
    dt_deque: deque[float] = deque(maxlen=200)
    ts_last = time.time()

    use_jbuf = buffer_ms > 0
    if use_jbuf:
        nominal_hz = 90.0
        target_depth = max(1, round(buffer_ms / 1000.0 * nominal_hz))
        jbuf: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque()
        jbuf_lock = threading.Lock()
        jbuf_filled = threading.Event()

        def _jitter_output():
            dt_out = 1.0 / nominal_hz
            out_qpos = None
            out_hand = np.zeros(4, dtype=np.float32)
            out_hand_qpos = np.zeros(len(buf_hand_qpos), dtype=np.float32)
            jbuf_filled.wait()
            if not ready_evt.is_set():
                ready_evt.set()
            while not stop_evt.is_set():
                popped = False
                with jbuf_lock:
                    if jbuf:
                        out_qpos, out_hand, out_hand_qpos = jbuf.popleft()
                        depth = len(jbuf)
                        popped = True
                    else:
                        depth = 0
                if popped and out_qpos is not None:
                    with buf_hand.get_lock():
                        np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = out_hand
                    with buf_hand_qpos.get_lock():
                        np.frombuffer(buf_hand_qpos.get_obj(), dtype=np.float32)[:] = out_hand_qpos
                    with buf.get_lock(), ts.get_lock():
                        np.frombuffer(buf.get_obj(), dtype=np.float32, count=out_qpos.size)[:] = out_qpos
                        ts.value = time.time()

                depth_err = depth - target_depth
                dt_out = (1.0 / nominal_hz) * (1.0 - 0.02 * depth_err)
                dt_out = max(0.005, min(0.030, dt_out))
                time.sleep(dt_out)

        threading.Thread(target=_jitter_output, daemon=True).start()
        print(f"[Retarget] BrainCo hand jitter buffer: {buffer_ms:.0f} ms ({target_depth} frames)")

    try:
        while not stop_evt.is_set():
            frame = get_frame()
            if frame is None:
                continue

            l_open, l_dist = _detect_hand_open(frame, **_HAND_JOINTS["left"])
            r_open, r_dist = _detect_hand_open(frame, **_HAND_JOINTS["right"])
            hand_data = np.array([float(l_open), l_dist, float(r_open), r_dist], dtype=np.float32)

            try:
                hand_qpos = _retarget_noitom_hand_qpos(frame, hand_target)
            except (KeyError, TypeError, ValueError) as e:
                print(f"[Retarget] BrainCo hand retarget error: {e}")
                hand_qpos = np.zeros(len(buf_hand_qpos), dtype=np.float32)

            if not use_jbuf:
                with buf_hand.get_lock():
                    np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand_data
                with buf_hand_qpos.get_lock():
                    np.frombuffer(buf_hand_qpos.get_obj(), dtype=np.float32)[:] = hand_qpos

            try:
                qpos = retarget.retarget(frame)
            except Exception as e:
                import traceback

                print(f"[Retarget] error: {e}\n{traceback.format_exc()}")
                continue

            if qpos_last is not None:
                qpos = qpos_last * ema_alpha + qpos * (1.0 - ema_alpha)
            qpos_last = qpos.copy()
            qpos = np.asarray(qpos, dtype=np.float32)

            if use_jbuf:
                with jbuf_lock:
                    jbuf.append((qpos.copy(), hand_data, hand_qpos.copy()))
                    if not jbuf_filled.is_set() and len(jbuf) >= target_depth:
                        jbuf_filled.set()
                    while len(jbuf) > target_depth * 3:
                        jbuf.popleft()
            else:
                with buf.get_lock(), ts.get_lock():
                    mv = np.frombuffer(buf.get_obj(), dtype=np.float32, count=qpos.size)
                    mv[:] = qpos
                    ts.value = time.time()

            ts_now = time.time()
            dt_deque.append(ts_now - ts_last)
            ts_last = ts_now
            if len(dt_deque) == dt_deque.maxlen:
                dt_deque.clear()

            if not use_jbuf and not ready_evt.is_set():
                ready_evt.set()
    finally:
        if (mocap_type or "").lower() == "pnlink" and hasattr(client, "stop"):
            client.stop()


def start_realtime_retarget_with_brainco_hands(
    robot: str = "unitree_g1",
    dof_full: int = 36,
    actual_human_height: float = 1.6,
    visualize_retarget: bool = True,
    mocap_type: str = "pnlink",
    buffer_ms: float = 0.0,
    hand_target: str = "brainco2",
    rt_pin: tuple[int, int] | None = None,
):
    if hand_target not in _BRAINCO_TARGETS:
        raise ValueError(f"hand_target must be one of {_BRAINCO_TARGETS}, got {hand_target!r}")

    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", dof_full, lock=True)
    buf_hand = ctx.Array("f", 4, lock=True)
    buf_hand_qpos = ctx.Array("f", 24, lock=True)
    ts = ctx.Value("d", 0.0)
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()

    p = ctx.Process(
        target=_retarget_worker_with_brainco_hands,
        args=(
            buf,
            buf_hand,
            buf_hand_qpos,
            ts,
            ready_evt,
            stop_evt,
            robot,
            actual_human_height,
            mocap_type,
            buffer_ms,
            hand_target,
            rt_pin,
        ),
        daemon=True,
    )
    p.start()

    vis_p = None
    if visualize_retarget:
        vis_p = ctx.Process(
            target=_visualize_worker,
            args=(buf, stop_evt, robot),
            daemon=True,
        )
        vis_p.start()

    sess = {
        "proc": p,
        "vis_proc": vis_p,
        "ready_evt": ready_evt,
        "stop_evt": stop_evt,
        "buf": buf,
        "buf_hand": buf_hand,
        "buf_hand_qpos": buf_hand_qpos,
        "ts": ts,
    }
    _RETARGET_SESSIONS.append(sess)
    atexit.register(stop_all_retarget_with_brainco_hands)

    return buf, ts, buf_hand, buf_hand_qpos


def read_brainco_hand_qpos_buffer(buf_hand_qpos) -> np.ndarray | None:
    if buf_hand_qpos is None:
        return None
    with buf_hand_qpos.get_lock():
        return np.frombuffer(buf_hand_qpos.get_obj(), dtype=np.float32).copy()


def stop_all_retarget_with_brainco_hands() -> None:
    for sess in _RETARGET_SESSIONS:
        try:
            sess["stop_evt"].set()
        except Exception:
            pass
    for sess in _RETARGET_SESSIONS:
        for key in ("proc", "vis_proc"):
            proc = sess.get(key)
            if proc is not None and proc.is_alive():
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
    _RETARGET_SESSIONS.clear()


# ---------------------------------------------------------------------------
# Offline reference loading with optional hand sidecar
# ---------------------------------------------------------------------------

def _body_qpos_from_possible_hand_qpos(qpos: np.ndarray) -> np.ndarray:
    """Trim full body+hand qpos to the 36-D tracker body qpos when needed."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] <= 36:
        return qpos

    joints = qpos[:, 7:]
    if joints.shape[1] >= 53:
        body = np.zeros((qpos.shape[0], 29), dtype=np.float32)
        body[:, :22] = joints[:, :22]
        body[:, 22:29] = joints[:, 34:41]
        return np.concatenate([qpos[:, :7], body], axis=1)

    return qpos[:, :36]


def _qpos2ctl_batch(hand_qpos: np.ndarray) -> np.ndarray:
    hand_qpos = np.asarray(hand_qpos, dtype=np.float32)
    ctrl = np.zeros((hand_qpos.shape[0], 6), dtype=np.float32)
    ctrl[:, 1] = hand_qpos[:, 8] / 1.5184
    ctrl[:, 0] = (hand_qpos[:, 9] + hand_qpos[:, 10]) / (1.0472 + 1.0472)
    ctrl[:, 2] = (hand_qpos[:, 6] + hand_qpos[:, 7]) / (1.4661 + 1.6930)
    ctrl[:, 3] = (hand_qpos[:, 4] + hand_qpos[:, 5]) / (1.4661 + 1.6930)
    ctrl[:, 4] = (hand_qpos[:, 2] + hand_qpos[:, 3]) / (1.4661 + 1.6930)
    ctrl[:, 5] = (hand_qpos[:, 0] + hand_qpos[:, 1]) / (1.4661 + 1.6930)
    return np.clip(ctrl, 0.0, 1.0)


def _hand_cmd_from_full_joint_angles(joints: np.ndarray, scale: float) -> np.ndarray | None:
    """Parse 53-D body+hand joint arrays used by the arm-sdk deployment script."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] < 53:
        return None

    left_hand_qpos = np.zeros((joints.shape[0], 12), dtype=np.float32)
    right_hand_qpos = np.zeros((joints.shape[0], 12), dtype=np.float32)

    left_hand_qpos[:, 0:2] = joints[:, 32:34]
    left_hand_qpos[:, 2:4] = joints[:, 30:32]
    left_hand_qpos[:, 4:6] = joints[:, 28:30]
    left_hand_qpos[:, 6:8] = joints[:, 26:28]
    left_hand_qpos[:, 8:12] = joints[:, 22:26]

    right_hand_qpos[:, 0:2] = joints[:, 51:53]
    right_hand_qpos[:, 2:4] = joints[:, 49:51]
    right_hand_qpos[:, 4:6] = joints[:, 47:49]
    right_hand_qpos[:, 6:8] = joints[:, 45:47]
    right_hand_qpos[:, 8:12] = joints[:, 41:45]

    right_cmd = _qpos2ctl_batch(right_hand_qpos)
    left_cmd = _qpos2ctl_batch(left_hand_qpos)
    return np.clip(np.concatenate([right_cmd, left_cmd], axis=1) * float(scale), 0.0, 1.0)


def load_offline_motions_with_brainco_hands(
    track_dir: str,
    mj_model: mujoco.MjModel,
    freq: int = 50,
    hand_scale: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    folder = Path(track_dir)
    files = [folder] if folder.is_file() else sorted(folder.glob("*.npz"))

    motions = []
    hand_motions = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            print(f"[WARN] Skipping {f.name}: no qpos field")
            continue

        raw_qpos = np.asarray(data["qpos"], dtype=np.float32)
        data["qpos"] = apply_ema_qpos(_body_qpos_from_possible_hand_qpos(raw_qpos))
        freq_src = float(data.get("frequency", 50))
        kpt_data = qpos2kpt(
            mj_model,
            np.float32(data["qpos"]),
            freq_src=freq_src,
            freq_tgt=freq,
            interp_sec=0.5,
            end_default_sec=0.5,
            debug=False,
            foot_contact_est=False,
            height_clip_mode=None,
            video_path=None,
        )

        hand_cmd = None
        if raw_qpos.ndim == 2 and raw_qpos.shape[1] > 7:
            hand_cmd = _hand_cmd_from_full_joint_angles(raw_qpos[:, 7:], scale=hand_scale)

        motions.append({"data": kpt_data, "filename": f.name})
        hand_motions.append({"data": hand_cmd, "filename": f.name})
        hand_note = " + hands" if hand_cmd is not None else ""
        print(f"  Mode {len(motions)+1}: {f.name} ({len(kpt_data['qpos'])} frames{hand_note})")

    return motions, hand_motions


# ---------------------------------------------------------------------------
# Real-robot loop
# ---------------------------------------------------------------------------

def run_real(args: "NaoDeployArgs"):
    from deploy.brainco.brainco_controller import BraincoController
    from deploy.real_robot import KeyMap, LowLevelControlG1
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.utils.thread import RecurrentThread

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    policy_args = PolicyArgs(onnx_track=args.onnx_track)
    track_policy = get_policy_onnx(policy_args, use_trt=True, strict_trt=True)
    walk_policy = WalkPolicy(args.onnx_walk)

    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions, hand_motions = load_offline_motions_with_brainco_hands(
        args.track_dir,
        convert_model,
        freq,
        hand_scale=args.brainco_hand_scale,
    )

    pygame.init()
    pygame.display.quit()
    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)

    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = consts.load_mj_model(xml_path)
    phantom_model.opt.timestep = 0.001

    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    mocap_buffer = None
    buf_hand_qpos = None
    if not args.no_mocap:
        try:
            buf_mocap, ts_mocap, _, buf_hand_qpos = start_realtime_retarget_with_brainco_hands(
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type="pnlink",
                buffer_ms=args.buffer_ms,
                hand_target=args.hand_target,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] BrainCo retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start BrainCo retarget: {e}. Online mode disabled.")
    else:
        print("[Mocap] Disabled (--no-mocap). Online mode disabled.")

    hand_ctrl = None
    hand_smoother = BraincoHandSmoother(
        alpha=args.brainco_hand_smooth_alpha,
        scale=args.brainco_hand_scale,
    )
    try:
        hand_ctrl = BraincoController(fps=args.brainco_hand_fps)
        rest = hand_smoother.reset()
        hand_ctrl.set_action({"qpos": rest})
        print("[BrainCo] Hand controller ready.")
    except Exception as e:
        print(f"[BrainCo] Failed to init hand controller: {e}")
        hand_ctrl = None

    last_mode = 0
    track_step = 0
    ref_traj = None
    hand_traj = None
    prev_online_ref = None

    def set_hand_rest():
        if hand_ctrl is not None:
            hand_ctrl.set_action({"qpos": hand_smoother.reset()})

    def locomotion_step():
        nonlocal last_mode, track_step, ref_traj, hand_traj
        nonlocal prev_online_ref

        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        cmd = keyboard.step_command()
        mode = cmd.mode
        offline_frame_idx = None

        entering_track = (last_mode == 0) and (mode >= 1)
        leaving_track = (last_mode >= 1) and (mode == 0)

        if entering_track:
            infer_fn.info["last_action"][:] = 0
            live_converter.reset()
            hand_smoother.reset()
            prev_online_ref = None
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            live_converter.set_robot_initial_pose(root_quat, robot_xy)
            if mode >= 2:
                traj_idx = mode - 2
                if traj_idx < len(ref_motions):
                    ref_traj = ref_motions[traj_idx]["data"]
                    hand_traj = hand_motions[traj_idx]["data"]
                    track_step = 0
            elif mode == 1 and mocap_buffer is None:
                print("[Track] Online retarget unavailable (--no-mocap or init failed)")

        if leaving_track:
            live_converter.reset()
            hand_traj = None
            set_hand_rest()

        if mode == 0:
            cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
            motor_targets = walk_policy.infer(root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel)
            low_ctrl.step(motor_targets, KPs_walking, KDs_walking)
            if args.rest_hand_in_walk:
                set_hand_rest()
        else:
            if mode == 1:
                if mocap_buffer is None:
                    last_mode = mode
                    return
                qpos_full, _ = mocap_buffer.read()
                ref_new = live_converter.convert(qpos_full)
                ref_curr = ref_new if prev_online_ref is None else prev_online_ref
                ref_next = ref_new
                prev_online_ref = ref_new
            else:
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    frame_idx = min(track_step, traj_len - 1)
                    offline_frame_idx = frame_idx
                    ref_curr = jtu.tree_map(lambda x: x[frame_idx][None], ref_traj)
                    next_step = min(frame_idx + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    track_step = min(track_step + 1, traj_len - 1)
                else:
                    last_mode = mode
                    return

            motor_targets = infer_fn.infer_onnx_real(
                root_quat,
                root_gyro,
                jnt_qpos,
                jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs)

            if hand_ctrl is not None:
                if mode == 1:
                    hand_qpos = read_brainco_hand_qpos_buffer(buf_hand_qpos)
                    hand_cmd = hand_smoother.update_from_qpos24(hand_qpos)
                elif hand_traj is not None:
                    frame_idx = min(offline_frame_idx or 0, len(hand_traj) - 1)
                    hand_cmd = np.asarray(hand_traj[frame_idx], dtype=np.float32)
                    hand_smoother.last_cmd = hand_cmd.copy()
                else:
                    hand_cmd = hand_smoother.reset()
                hand_ctrl.set_action({"qpos": hand_cmd})

        last_mode = mode

    print("<Mode: Damping> Waiting for <start> on remote...")
    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        time.sleep(ctrl_dt)

    low_ctrl.move_to_default_pos(duration=2.0)
    set_hand_rest()
    print("<Mode: Default> Waiting for <A> on remote...")
    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        time.sleep(ctrl_dt)

    print("<Mode: Locomotion + BrainCo hands> Starting control loop...")
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
        set_hand_rest()
        low_ctrl.set_motor_damping()
        keyboard.close()
        stop_all_retarget_with_brainco_hands()
        print("[Real] Exited.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class NaoDeployArgs:
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    track_dir: str = "storage/test"
    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    convert_xml_path: str = str(consts.TRACK_XML)
    real: bool = False
    debug: bool = False
    freq: int = 50

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"
    human_height: float = 1.7
    visualize_retarget: bool = True
    buffer_ms: float = 30.0

    # Real robot
    net: str = "enx6c1ff76e8ef5"

    # BrainCo hand additions.  These are only used by this module's real path.
    hand_target: str = "brainco2"
    brainco_hand_fps: int = 100
    brainco_hand_smooth_alpha: float = 0.45
    brainco_hand_scale: float = 1.0
    rest_hand_in_walk: bool = True


def _to_play_track_args(args: NaoDeployArgs) -> PlayTrackArgs:
    return PlayTrackArgs(
        onnx_walk=args.onnx_walk,
        track_dir=args.track_dir,
        onnx_track=args.onnx_track,
        convert_xml_path=args.convert_xml_path,
        real=False,
        debug=args.debug,
        freq=args.freq,
        no_mocap=args.no_mocap,
        mocap_type=args.mocap_type,
        human_height=args.human_height,
        visualize_retarget=args.visualize_retarget,
        buffer_ms=args.buffer_ms,
        net=args.net,
    )


def main(args: NaoDeployArgs):
    if args.real:
        run_real(args)
    else:
        run_sim(_to_play_track_args(args))


if __name__ == "__main__":
    main(tyro.cli(NaoDeployArgs))
