"""Deployment entry for the Transformer tracker (sim and real robot).

Mirrors ``deploy/play_track.py`` but uses ``G1TrackTransformerInferFn`` so
the ONNX policy is driven by a rolling K-frame observation history.  Walk
policy, keyboard command, mocap online retarget, and offline tracking are
all reused from the dense-tracking deployment stack.

Modes (keyboard number keys):
    0 = Walk policy
    1 = Online retarget (Noitom / Xsens mocap)
    2+ = Offline tracking (reference trajectories from track_dir)

Usage::

    # Simulation (offline tracking)
    python -m projects.transformer.deploy \\
        --onnx-track storage/ckpt/transformer.onnx \\
        --policy-type transformer \\
        --track-dir storage/test \\
        --history-len 4

    # Real robot
    python -m projects.transformer.deploy \\
        --real \\
        --onnx-track storage/ckpt/transformer.onnx \\
        --policy-type transformer \\
        --history-len 4 \\
        --net enx6c1ff76e8ef5
"""

from __future__ import annotations

import os
import time
import signal
from dataclasses import dataclass

import tyro
import mujoco
import mujoco.viewer
import numpy as np
from jax import tree_util as jtu
from loop_rate_limiters import RateLimiter

from tracking import constants as consts
from utils.sim_mj import get_sensor_data as mj_sensor
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from tracking.infer_utils import G1TrackMjSim, g1_infer_env_config
from projects.transformer.infer_utils import G1TrackTransformerInferFn

from deploy.play_track import (
    MocapBuffer,
    LiveRefConverter,
    load_offline_motions,
    _print_offline_metrics,
)
from deploy.walk_policy import WalkPolicy
from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.constants import DEFAULT_QPOS as DEFAULT_QPOS_JOINT, KPs_walking, KDs_walking
from tracking.metrics import (
    calculate_kpt_mae_error,
    calculate_joint_tracking_error,
    calculate_root_tracking_error,
)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_sim(args: "DeployArgs"):
    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    policy_args = PolicyArgs(
        onnx_track=args.onnx_track, policy_type="transformer",
    )
    track_policy = get_policy_onnx(policy_args)
    walk_policy = WalkPolicy(args.onnx_walk)

    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    init_qpos = consts.DEFAULT_QPOS.copy()
    mj_sim = G1TrackMjSim(init_qpos=init_qpos, headless=False, ctrl_dt=ctrl_dt)

    infer_fn = G1TrackTransformerInferFn(
        env_cfg, mj_sim.mj_model, track_policy,
        privileged=args.privileged,
        history_len=args.history_len,
    )

    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    if not mj_sim.headless and mj_sim.viewer is not None:
        cam = mj_sim.viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.azimuth, cam.elevation, cam.distance = 90.0, -20.0, 2.0

    live_converter = LiveRefConverter(mj_sim.mj_model, ctrl_dt)

    mocap_buffer = None
    if not args.no_mocap:
        try:
            from deploy.retarget import start_realtime_retarget
            buf_mocap, ts_mocap, _ = start_realtime_retarget(
                robot="unitree_g1", dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=args.mocap_type, buffer_ms=args.buffer_ms,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")

    rate = RateLimiter(frequency=freq, warn=False)
    last_mode = 0
    track_step = 0
    ref_traj = None
    traj_metrics = None
    traj_filename = None
    prev_online_ref = None

    print(f"\n=== Transformer Tracking Simulation ready (history_len={args.history_len}). ===\n")

    try:
        while True:
            if keyboard.check_reset_request():
                state = mj_sim.reset(state)
                infer_fn.reset_history()
                infer_fn.info["step"] = 0
                live_converter.reset()
                print("[Reset] Simulation reset.")

            cmd = keyboard.step_command()
            mode = cmd.mode

            entering_track = (last_mode == 0) and (mode >= 1)
            leaving_track = (last_mode >= 1) and (mode == 0)

            if entering_track:
                infer_fn.reset_history()
                live_converter.reset()
                prev_online_ref = None
                if mode >= 2:
                    traj_idx = mode - 2
                    if traj_idx < len(ref_motions):
                        ref_traj = ref_motions[traj_idx]["data"]
                        track_step = 0
                        traj_filename = ref_motions[traj_idx]["filename"]
                        traj_metrics = {k: [] for k in (
                            "kpt_pos_errors", "kpt_rot_errors",
                            "joint_pos_errors", "joint_vel_errors",
                            "root_pos_errors", "root_vel_errors",
                            "root_yaw_errors", "state_history",
                        )}
                        print(f"[Track] Start offline: {traj_filename}")
                    else:
                        print(f"[Track] Invalid trajectory index {traj_idx}")
                        mode = 0
                elif mode == 1:
                    print("[Track] Start online retarget")

            if leaving_track:
                if last_mode >= 2 and traj_metrics and traj_metrics["state_history"]:
                    _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model, ctrl_dt)
                traj_metrics = None
                traj_filename = None
                live_converter.reset()
                infer_fn.reset_history()
                print("[Track] Back to walk mode")

            if mode == 0:
                cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
                gyro = mj_sensor(mj_sim.mj_model, state.mj_data, "gyro_pelvis")
                motor_targets = walk_policy.infer(
                    state.mj_data.qpos[3:7], gyro,
                    state.mj_data.qpos[7:], state.mj_data.qvel[6:],
                    cmd_vel,
                )
                for _ in range(mj_sim.num_sim_substeps):
                    torques = KPs_walking * (motor_targets - state.mj_data.qpos[7:]) + KDs_walking * (-state.mj_data.qvel[6:])
                    torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
                    state.mj_data.ctrl[:] = torques
                    mujoco.mj_step(mj_sim.mj_model, state.mj_data)

            elif mode == 1:
                if mocap_buffer is not None:
                    qpos_full, _ = mocap_buffer.read()
                    ref_new = live_converter.convert(qpos_full)
                    ref_curr = ref_new if prev_online_ref is None else prev_online_ref
                    ref_next = ref_new
                    prev_online_ref = ref_new
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

            else:
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

                    if traj_metrics is not None:
                        kpt_p, kpt_r = calculate_kpt_mae_error(state, ref_curr, ref_next, mj_sim.mj_model)
                        jnt_p, jnt_v = calculate_joint_tracking_error(state, ref_curr)
                        rp, rv, ry = calculate_root_tracking_error(state, ref_curr)
                        traj_metrics["kpt_pos_errors"].append(kpt_p)
                        traj_metrics["kpt_rot_errors"].append(kpt_r)
                        traj_metrics["joint_pos_errors"].append(jnt_p)
                        traj_metrics["joint_vel_errors"].append(jnt_v)
                        traj_metrics["root_pos_errors"].append(rp)
                        traj_metrics["root_vel_errors"].append(rv)
                        traj_metrics["root_yaw_errors"].append(ry)
                        traj_metrics["state_history"].append({
                            "qpos": state.mj_data.qpos.copy(),
                            "qvel": state.mj_data.qvel.copy(),
                            "xpos": state.mj_data.xpos.copy(),
                            "xmat": state.mj_data.xmat.copy(),
                        })

                    track_step += 1
                    if track_step >= traj_len and traj_metrics and traj_metrics["state_history"]:
                        _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model, ctrl_dt)
                        traj_metrics = None
                        traj_filename = None
                        ref_traj = None

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
    from deploy.hand_control import Dex3Controller, update_hand_from_mocap
    from deploy.retarget import start_realtime_retarget, read_hand_buffer

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    policy_args = PolicyArgs(
        onnx_track=args.onnx_track, policy_type="transformer",
    )
    track_policy = get_policy_onnx(policy_args, use_trt=True, strict_trt=True)
    walk_policy = WalkPolicy(args.onnx_walk)

    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    pygame.init()
    pygame.display.quit()
    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)

    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = mujoco.MjModel.from_xml_path(xml_path)
    phantom_model.opt.timestep = 0.001

    infer_fn = G1TrackTransformerInferFn(
        env_cfg, phantom_model, track_policy,
        privileged=args.privileged,
        history_len=args.history_len,
    )
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    mocap_buffer = None
    buf_hand = None
    if not args.no_mocap:
        try:
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                robot="unitree_g1", dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=args.mocap_type, buffer_ms=args.buffer_ms,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")
    else:
        print("[Mocap] Disabled (--no-mocap). Online mode disabled.")

    hand_ctrl = None
    if args.enable_hand:
        try:
            hand_ctrl = Dex3Controller(net=args.net, re_init=False)
        except Exception as e:
            print(f"[Hand] Failed to init: {e}")

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
            infer_fn.reset_history()
            live_converter.reset()
            prev_online_ref = None
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
            infer_fn.reset_history()

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
                ref_curr = ref_new if prev_online_ref is None else prev_online_ref
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

    print("<Mode: Damping> Waiting for <start> on remote...")
    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        time.sleep(ctrl_dt)

    low_ctrl.move_to_default_pos(duration=2.0)
    print("<Mode: Default> Waiting for <A> on remote...")
    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        time.sleep(ctrl_dt)

    print(f"<Mode: Locomotion (Transformer, history_len={args.history_len})> Starting control loop...")
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
    onnx_track: str = "storage/ckpts/transformer.onnx"
    convert_xml_path: str = str(consts.TRACK_XML)
    real: bool = False
    debug: bool = False
    freq: int = 50
    privileged: bool = False
    history_len: int = 4

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"
    human_height: float = 1.7
    visualize_retarget: bool = True
    buffer_ms: float = 30.0

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
