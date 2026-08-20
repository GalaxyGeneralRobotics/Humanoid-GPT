"""Onboard tracking on Unitree G1 that *receives* retargeted frames over WiFi.

This is the cable-free, low-CPU variant of
``deploy.onboard_deploy.play_track_onboard``.  GMR (motion retargeting) and
the Noitom PNLink client are pushed off to a workstation that runs
``deploy.onboard_deploy_wo_GMR.host_sender``; this process only does what
the Jetson is actually good at:

    UDP listener (one tiny socket thread, ~180 B/pkt)
            │
            ▼
    Latest retargeted frame slot  (lock-protected, single-slot "queue")
            │
            ▼
    LiveRefConverter (MuJoCo FK, ~0.3 ms)
            │
            ▼
    ONNX/TRT tracking policy
            │
            ▼
    DDS motor commands (eth0 internal bus)

Compared with ``play_track_onboard.py`` this saves the Jetson the cost of
running NoitomClient + GMR (combined ~25-40 ms wall time per frame at 90 Hz),
freeing CPU budget for tighter 50 Hz control loops.  The trade-off is that
the workstation and the G1 must share a WiFi network with low loss; UDP is
unreliable on purpose so that occasional drops never block control.

Usage on the G1 (over SSH)::

    # On the 4090 workstation:
    python -m deploy.onboard_deploy_wo_GMR.host_sender --robot-ip <g1_wifi_ip>

    # On the G1:
    python -m deploy.onboard_deploy_wo_GMR.play_track_onboard_wo_GMR \\
        --onnx-track storage/ckpts/pns_wo_priv264.onnx

Modes are the same as ``play_track_onboard``:
    0 = Walk policy
    1 = Online retarget (now sourced from the network instead of GMR)
    2+ = Offline trajectory tracking
"""

from __future__ import annotations

# Mirror the aarch64 + mujoco-warp dance from play_track_onboard so this file
# can be launched standalone without first importing the sibling module.
import ctypes
import platform
if platform.machine() == "aarch64":
    for _lib in ["/lib/aarch64-linux-gnu/libGLdispatch.so.0"]:
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    import site
    from pathlib import Path as _Path
    for _sp in site.getsitepackages():
        for _p in _Path(_sp).glob("torch.libs/libgomp-*.so*"):
            try:
                ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break

import os
import sys
import time
import tyro


class _MujocoWarpFilter:
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        if "Failed to import warp" not in msg and "Failed to import mujoco_warp" not in msg:
            self._stream.write(msg)
    def flush(self):
        self._stream.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _MujocoWarpFilter(sys.stdout)
import mujoco  # noqa: E402
sys.stdout = sys.stdout._stream
del _MujocoWarpFilter

import curses
import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking import constants as consts
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config

from deploy.constants import (
    DEFAULT_QPOS as DEFAULT_QPOS_JOINT,
    KPs_walking,
    KDs_walking,
)
from deploy.walk_policy import WalkPolicy

# Re-use the heavy lifting from the sibling onboard deployment.  These are
# pure helpers / dataclasses with no Noitom or GMR dependency, so importing
# them on the G1 in the cable-free configuration is safe.
from deploy.onboard_deploy.play_track_onboard import (
    HighCommand,
    LiveRefConverter,
    TerminalCMD,
    _tree_index,
    check_mocap_health,
    load_offline_motions,
)

from .net_recv import NetMocapReceiver
from .protocol import DEFAULT_PORT, G1_DOF_FULL


# ---------------------------------------------------------------------------
# Logging (separate file from play_track_onboard so we don't shadow its log)
# ---------------------------------------------------------------------------

_LOG_DIR = Path("/tmp/humanoid_gpt_onboard")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"deploy_wo_gmr_{time.strftime('%Y%m%d_%H%M%S')}.log"
# Use a dedicated handler so we don't fight basicConfig set elsewhere.
_log = logging.getLogger("onboard_wo_gmr")
_log.setLevel(logging.INFO)
_log.propagate = False
if not _log.handlers:
    _h = logging.FileHandler(str(_LOG_FILE))
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_h)


# ---------------------------------------------------------------------------
# Main onboard deployment loop (cable-free)
# ---------------------------------------------------------------------------


def _run_onboard(stdscr, args: "OnboardWoGmrArgs"):
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.curs_set(0)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.utils.thread import RecurrentThread
    from deploy.real_robot import LowLevelControlG1, KeyMap
    from deploy.hand_control import Dex3Controller, update_hand_from_mocap

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    # Policies
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
    )
    track_policy = get_policy_onnx(
        policy_args, use_trt=args.use_trt, strict_trt=False
    )
    walk_policy = WalkPolicy(args.onnx_walk)

    # Offline reference motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    _log.info("Loading offline reference motions from %s", args.track_dir)
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)
    for i, m in enumerate(ref_motions):
        _log.info("  Mode %d: %s (%d frames)",
                  i + 2, m["filename"], len(m["data"]["qpos"]))

    keyboard = TerminalCMD(num_track_ref=len(ref_motions))

    # DDS channel + low-level motor controller
    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)
    keyboard.set_status(robot_status="Connected")
    _log.info("Robot connected via DDS on interface '%s'", args.net)

    # Phantom MuJoCo model used only for reference FK
    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = mujoco.MjModel.from_xml_path(xml_path)
    phantom_model.opt.timestep = 0.001
    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # Network mocap receiver: runs in a dedicated subprocess + shared
    # memory, mirroring how GMR is launched on the workstation-assisted
    # variant.  Keeping the UDP recv loop out of the main interpreter
    # eliminates GIL contention with the 50 Hz control thread (which
    # showed up as motor "click" jitter in the threaded implementation).
    net_recv: NetMocapReceiver | None = None
    if not args.no_mocap:
        try:
            net_recv = NetMocapReceiver(
                host=args.listen_ip, port=args.listen_port,
                dof_full=G1_DOF_FULL,
                rt_pin=args.net_recv_rt_pin,
            )
            net_recv.start()
            keyboard.set_status(
                mocap_status=f"Listening {args.listen_ip}:{args.listen_port}"
            )
        except OSError as e:
            keyboard.set_status(mocap_status=f"Bind failed: {e}")
            _log.error("UDP bind failed: %s", e)
            net_recv = None

    # Hand controller (optional, driven by the hand field of incoming packets)
    hand_ctrl = None
    if args.enable_hand:
        try:
            hand_ctrl = Dex3Controller(net=args.net, re_init=False)
        except Exception as e:
            _log.error("Hand init failed: %s", e)

    # -- shared mutable state for the control thread -------------------------
    last_mode = 0
    track_step = 0
    ref_traj = None
    last_left_hand = None
    last_right_hand = None
    prev_online_ref = None
    _freq = {"n": 0, "t0": time.time(), "total": 0}
    _net_stats_t0 = time.time()
    _net_stats_last_recv = 0

    def locomotion_step():
        nonlocal last_mode, track_step, ref_traj
        nonlocal last_left_hand, last_right_hand, prev_online_ref
        nonlocal _net_stats_t0, _net_stats_last_recv

        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        cmd = keyboard.step_command()
        mode = cmd.mode

        entering = (last_mode == 0) and (mode >= 1)
        leaving = (last_mode >= 1) and (mode == 0)

        if entering:
            infer_fn.info["last_action"][:] = 0
            live_converter.reset()
            prev_online_ref = None
            robot_xy = np.array([0.0, 0.0], dtype=np.float32)
            live_converter.set_robot_initial_pose(root_quat, robot_xy)
            if mode >= 2:
                traj_idx = mode - 2
                if traj_idx < len(ref_motions):
                    ref_traj = ref_motions[traj_idx]["data"]
                    track_step = 0
                    tlen = len(ref_traj["qpos"])
                    keyboard.set_status(
                        track_info=f"{ref_motions[traj_idx]['filename']}  0/{tlen}"
                    )
                    _log.info("Offline track start: %s",
                              ref_motions[traj_idx]["filename"])

        if leaving:
            live_converter.reset()
            keyboard.set_status(track_info="")

        if mode == 0:
            cmd_vel = np.array(
                [cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32
            )
            motor_targets = walk_policy.infer(
                root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel
            )
            low_ctrl.step(motor_targets, KPs_walking, KDs_walking)

            if net_recv is not None:
                try:
                    qpos_full, _ = net_recv.read()
                    health = check_mocap_health(qpos_full)
                    keyboard.set_status(mocap_health=health)
                except Exception:
                    keyboard.set_status(mocap_health="READ ERR")

        else:
            if mode == 1:
                if net_recv is None:
                    last_mode = mode
                    return
                qpos_full, _ = net_recv.read()
                ref_new = live_converter.convert(qpos_full)
                if prev_online_ref is None:
                    ref_curr = ref_new
                else:
                    ref_curr = prev_online_ref
                ref_next = ref_new
                prev_online_ref = ref_new
            else:
                if ref_traj is None:
                    last_mode = mode
                    return
                traj_len = len(ref_traj["qpos"])
                ref_curr = _tree_index(ref_traj, track_step)
                nxt = min(track_step + 1, traj_len - 1)
                ref_next = _tree_index(ref_traj, nxt)
                track_step = min(track_step + 1, traj_len - 1)
                keyboard.set_status(track_info=f"{track_step}/{traj_len}")

            motor_targets = infer_fn.infer_onnx_real_fast(
                root_quat, root_gyro, jnt_qpos, jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(
                np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs
            )

            if net_recv is not None and hand_ctrl is not None:
                hand_cmd = net_recv.read_hand()
                last_left_hand, last_right_hand = update_hand_from_mocap(
                    hand_ctrl, hand_cmd, last_left_hand, last_right_hand
                )

        last_mode = mode

        _freq["n"] += 1
        _freq["total"] += 1
        elapsed = time.time() - _freq["t0"]
        if elapsed >= 1.0:
            keyboard.set_status(
                freq_hz=_freq["n"] / elapsed, total_steps=_freq["total"]
            )
            _freq["n"] = 0
            _freq["t0"] = time.time()

        # Network stats once per second (no curses redraw, just status row).
        if net_recv is not None:
            now = time.time()
            if now - _net_stats_t0 >= 1.0:
                s = net_recv.stats()
                hz = (s["recv"] - _net_stats_last_recv) / (now - _net_stats_t0)
                lag_ms = max(0.0, (now - s["last_send_ts"]) * 1e3) if s["last_send_ts"] > 0 else 0.0
                keyboard.set_status(
                    mocap_status=(
                        f"Net {hz:5.1f}Hz lag={lag_ms:5.1f}ms "
                        f"loss={s['missing']} ooo={s['ooo']}"
                    )
                )
                _net_stats_t0 = now
                _net_stats_last_recv = s["recv"]

    # ---- Startup sequence --------------------------------------------------

    if net_recv is not None and not args.no_mocap:
        keyboard.set_status(mocap_status="Waiting for first packet...")
        keyboard.draw(stdscr)
        if not net_recv.wait_first(args.startup_timeout_sec):
            keyboard.set_status(
                mocap_status=(
                    f"NO DATA (check host_sender + WiFi {args.listen_ip}:{args.listen_port})"
                )
            )
            _log.warning(
                "No mocap packet within %.1fs; tracking modes will use stale qpos",
                args.startup_timeout_sec,
            )
        else:
            keyboard.set_status(mocap_status="Network mocap connected")
            _log.info("First mocap packet received")

    keyboard.set_status(robot_status="Damping - press [start] on remote")
    keyboard.draw(stdscr)
    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        keyboard.draw(stdscr)
        time.sleep(ctrl_dt)

    keyboard.set_status(robot_status="Standing up...")
    keyboard.draw(stdscr)
    low_ctrl.move_to_default_pos(duration=2.0)

    keyboard.set_status(robot_status="Ready - press [A] on remote")
    keyboard.draw(stdscr)
    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        keyboard.draw(stdscr)
        time.sleep(ctrl_dt)

    # ---- Main control loop -------------------------------------------------

    try:
        os.sched_setaffinity(0, {4})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        _log.info("RT scheduling: core 4, SCHED_FIFO priority 50")
    except (OSError, PermissionError) as e:
        _log.warning("RT scheduling unavailable (%s), using default scheduler", e)

    gc.collect()
    gc.disable()
    keyboard.set_status(robot_status="Running")
    _log.info("Locomotion control loop started")

    loco_thread = RecurrentThread(
        interval=ctrl_dt, target=locomotion_step, name="loco"
    )
    loco_thread.Start()

    try:
        while True:
            keyboard.poll_key(stdscr)
            keyboard.draw(stdscr)
            time.sleep(0.02)

            if keyboard.step_command().kill:
                _log.info("Kill command received")
                break
            if low_ctrl.remote.button[KeyMap.select] == 1:
                _log.info("Remote [select] - emergency stop")
                break
    except KeyboardInterrupt:
        _log.info("Ctrl+C received")
    finally:
        gc.enable()
        try:
            low_ctrl.set_motor_damping()
        except Exception as e:
            _log.error("damping failed: %s", e)
        if net_recv is not None:
            net_recv.stop()
        keyboard.close()
        _log.info("Shutdown complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class OnboardWoGmrArgs:
    """Onboard deploy on Unitree G1 with retargeting offloaded to a workstation."""

    # Policy paths
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    track_dir: str = "storage/test"
    convert_xml_path: str = str(consts.TRACK_XML)

    # Robot / control
    net: str = "eth0"
    """DDS interface on the G1 (internal motor bus, typically eth0)."""
    freq: int = 50
    debug: bool = False
    use_trt: bool = True

    # Network mocap (replaces Noitom + GMR locally)
    no_mocap: bool = False
    """If set, skip the UDP listener and only run walk + offline modes."""
    listen_ip: str = "0.0.0.0"
    """Bind address for the mocap UDP listener (usually 0.0.0.0)."""
    listen_port: int = DEFAULT_PORT
    """UDP port the host_sender targets (see protocol.DEFAULT_PORT)."""
    startup_timeout_sec: float = 10.0
    """How long to wait for the first packet before starting up anyway.
    Tracking modes will refuse to act if no packets have arrived yet."""

    # Real-time scheduling for the UDP receiver subprocess.  Matches the
    # slot used by GMR in the sibling onboard deploy so the recv worker
    # stays off core 4 (which is reserved for the loco thread) and gets
    # priority over normal user tasks.
    net_recv_rt_pin: tuple[int, int] | None = (2, 40)
    """Pin UDP recv subprocess to (cpu_id, SCHED_FIFO priority).  Set to
    None to disable when running without CAP_SYS_NICE."""

    # Hand
    enable_hand: bool = False


def main(args: OnboardWoGmrArgs):
    curses.wrapper(lambda stdscr: _run_onboard(stdscr, args))


if __name__ == "__main__":
    main(tyro.cli(OnboardWoGmrArgs))
