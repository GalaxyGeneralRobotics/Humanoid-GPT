"""Cable-free onboard tracking on Unitree G1 with BrainCo dex-hand control.

Receiver-side counterpart to
``deploy.onboard_deploy_wo_GMR.host_sender --enable-brainco-hand``.  All of
the heavy work (NoitomClient + GMR body retarget + BrainCo hand retarget) is
performed on the 4090 workstation; this process only:

    UDP listener
            |
            v
    Latest body qpos + 24-D BrainCo hand qpos
            |
            v
    LiveRefConverter (MuJoCo FK)        BraincoHandSmoother (EMA)
            |                                       |
            v                                       v
    ONNX/TRT tracking policy                BraincoController DDS

Compared with the workstation's ``play_track_brainco.py``:

- We never spawn NoitomClient or GMR on the G1.
- We deliberately do *not* import ``deploy.play_track`` or
  ``deploy.brainco.play_track_brainco`` -- both have heavy top-level imports
  (pygame / jax / loop_rate_limiters).  Instead, the small BrainCo helper
  functions are duplicated below (~80 lines).  ``deploy.brainco`` and
  ``deploy.onboard_deploy_wo_GMR`` therefore stay independent.

Usage on the G1 (over SSH)::

    # On the 4090 workstation:
    python -m deploy.onboard_deploy_wo_GMR.host_sender \
        --robot-ip <g1_wifi_ip> --enable-brainco-hand

    # On the G1:
    python -m deploy.onboard_deploy_wo_GMR.play_track_onboard_wo_GMR_brainco \
        --onnx-track storage/ckpts/pns_wo_priv264.onnx

Modes mirror the BrainCo workstation deployment:
    0 = Walk policy (hands optionally driven back to rest)
    1 = Online retarget (body + BrainCo hand, both sourced from the network)
    2+ = Offline trajectories (body + per-frame 12-D BrainCo cmd)
"""

from __future__ import annotations

# Mirror the aarch64 + mujoco-warp dance from play_track_onboard so this
# file can be launched standalone without first importing siblings.
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
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import (
    G1TrackInferFn,
    apply_ema_qpos,
    g1_infer_env_config,
)
from tracking.policy import Args as PolicyArgs, get_policy_onnx

from deploy.constants import (
    DEFAULT_QPOS as DEFAULT_QPOS_JOINT,
    KPs_walking,
    KDs_walking,
)
from deploy.walk_policy import WalkPolicy

# Reuse low-overhead utilities from the Dex3 onboard build.
from deploy.onboard_deploy.play_track_onboard import (
    LiveRefConverter,
    TerminalCMD,
    _tree_index,
    check_mocap_health,
)

from .net_recv import NetMocapReceiver
from .protocol import DEFAULT_PORT, G1_DOF_FULL


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_DIR = Path("/tmp/humanoid_gpt_onboard")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"deploy_wo_gmr_brainco_{time.strftime('%Y%m%d_%H%M%S')}.log"
_log = logging.getLogger("onboard_wo_gmr_brainco")
_log.setLevel(logging.INFO)
_log.propagate = False
if not _log.handlers:
    _h = logging.FileHandler(str(_LOG_FILE))
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_h)


# ---------------------------------------------------------------------------
# BrainCo helpers (intentionally duplicated from
# deploy.brainco.play_track_brainco to keep the on-board build free of
# pygame / jax / loop_rate_limiters top-level imports).  If you fix a bug
# here, also fix it there -- the math is identical.
# ---------------------------------------------------------------------------

_HAND_CMD_DOF = 12


def _brainco_hand12_to_ctrl6(hand_qpos: np.ndarray) -> np.ndarray:
    """One 12-D BrainCo hand qpos -> 6 actuator commands.

    Source layout (GMR brainco target):
        thumb(4), index(2), middle(2), ring(2), pinky(2)
    Target layout (BraincoController):
        thumb, thumb_aux, index, middle, ring, pinky
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
    """24-D retargeted qpos (left12 then right12) -> 12-D BraincoController cmd
    (right6 then left6)."""
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
    """Parse 53-D body+hand joint arrays used by the arm-sdk deployment."""
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

    motions: list[dict] = []
    hand_motions: list[dict] = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            _log.warning("Skipping %s: no qpos field", f.name)
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

    return motions, hand_motions


# ---------------------------------------------------------------------------
# Main onboard deployment loop (cable-free, BrainCo hand)
# ---------------------------------------------------------------------------


def _run_onboard(stdscr, args: "OnboardWoGmrBraincoArgs"):
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
    from deploy.brainco.brainco_controller import BraincoController

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

    # Offline motions (with optional per-frame BrainCo hand cmd)
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    _log.info("Loading offline reference motions from %s", args.track_dir)
    ref_motions, hand_motions = load_offline_motions_with_brainco_hands(
        args.track_dir, convert_model, freq, hand_scale=args.brainco_hand_scale,
    )
    for i, m in enumerate(ref_motions):
        has_hands = hand_motions[i]["data"] is not None
        _log.info("  Mode %d: %s (%d frames%s)",
                  i + 2, m["filename"], len(m["data"]["qpos"]),
                  " + hands" if has_hands else "")

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

    # Network mocap receiver (carries 24-D BrainCo qpos in addition to
    # body).  Runs in a dedicated subprocess + shared memory so GIL
    # contention with the 50 Hz loco thread (the source of motor "click"
    # jitter in the threaded build) is eliminated.
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

    # BrainCo hand controller
    hand_ctrl: BraincoController | None = None
    hand_smoother = BraincoHandSmoother(
        alpha=args.brainco_hand_smooth_alpha,
        scale=args.brainco_hand_scale,
    )
    if args.enable_brainco_hand:
        try:
            hand_ctrl = BraincoController(fps=args.brainco_hand_fps)
            hand_ctrl.set_action({"qpos": hand_smoother.reset()})
            _log.info("BrainCo hand controller ready (fps=%d).", args.brainco_hand_fps)
        except Exception as e:
            _log.error("BrainCo init failed: %s", e)
            hand_ctrl = None

    def set_hand_rest():
        if hand_ctrl is not None:
            hand_ctrl.set_action({"qpos": hand_smoother.reset()})

    # -- shared mutable state for the control thread -------------------------
    last_mode = 0
    track_step = 0
    ref_traj = None
    hand_traj: np.ndarray | None = None
    prev_online_ref = None
    _freq = {"n": 0, "t0": time.time(), "total": 0}
    _net_stats_t0 = time.time()
    _net_stats_last_recv = 0

    def locomotion_step():
        nonlocal last_mode, track_step, ref_traj, hand_traj
        nonlocal prev_online_ref, _net_stats_t0, _net_stats_last_recv

        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        cmd = keyboard.step_command()
        mode = cmd.mode
        offline_frame_idx: int | None = None

        entering = (last_mode == 0) and (mode >= 1)
        leaving = (last_mode >= 1) and (mode == 0)

        if entering:
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
                    tlen = len(ref_traj["qpos"])
                    keyboard.set_status(
                        track_info=f"{ref_motions[traj_idx]['filename']}  0/{tlen}"
                    )
                    _log.info("Offline track start: %s",
                              ref_motions[traj_idx]["filename"])

        if leaving:
            live_converter.reset()
            hand_traj = None
            keyboard.set_status(track_info="")
            set_hand_rest()

        if mode == 0:
            cmd_vel = np.array(
                [cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32
            )
            motor_targets = walk_policy.infer(
                root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel
            )
            low_ctrl.step(motor_targets, KPs_walking, KDs_walking)

            if args.rest_hand_in_walk:
                set_hand_rest()

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
                ref_curr = ref_new if prev_online_ref is None else prev_online_ref
                ref_next = ref_new
                prev_online_ref = ref_new
            else:
                if ref_traj is None:
                    last_mode = mode
                    return
                traj_len = len(ref_traj["qpos"])
                frame_idx = min(track_step, traj_len - 1)
                offline_frame_idx = frame_idx
                ref_curr = _tree_index(ref_traj, frame_idx)
                ref_next = _tree_index(ref_traj, min(frame_idx + 1, traj_len - 1))
                track_step = min(track_step + 1, traj_len - 1)
                keyboard.set_status(track_info=f"{track_step}/{traj_len}")

            motor_targets = infer_fn.infer_onnx_real_fast(
                root_quat, root_gyro, jnt_qpos, jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(
                np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs
            )

            if hand_ctrl is not None:
                if mode == 1:
                    bq = net_recv.read_brainco_qpos() if net_recv is not None else None
                    hand_cmd = hand_smoother.update_from_qpos24(bq)
                elif hand_traj is not None:
                    frame_idx = min(offline_frame_idx or 0, len(hand_traj) - 1)
                    hand_cmd = np.asarray(hand_traj[frame_idx], dtype=np.float32)
                    hand_smoother.last_cmd = hand_cmd.copy()
                else:
                    hand_cmd = hand_smoother.reset()
                hand_ctrl.set_action({"qpos": hand_cmd})

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
                    f"NO DATA (check host_sender + WiFi "
                    f"{args.listen_ip}:{args.listen_port})"
                )
            )
            _log.warning(
                "No mocap packet within %.1fs; tracking modes will use stale qpos",
                args.startup_timeout_sec,
            )
        else:
            # Warn if the host did not include BrainCo qpos.
            if hand_ctrl is not None and net_recv.read_brainco_qpos() is None:
                keyboard.set_status(
                    mocap_status="WARN: host did not send brainco_qpos"
                )
                _log.warning(
                    "First packet contains no BrainCo qpos -- check that "
                    "host_sender was launched with --enable-brainco-hand"
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
    set_hand_rest()

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
    _log.info("Locomotion + BrainCo control loop started")

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
            set_hand_rest()
        except Exception as e:
            _log.error("hand rest failed: %s", e)
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
class OnboardWoGmrBraincoArgs:
    """Cable-free onboard deploy with BrainCo dex-hand control."""

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

    # Network mocap
    no_mocap: bool = False
    """If set, skip the UDP listener and only run walk + offline modes."""
    listen_ip: str = "0.0.0.0"
    listen_port: int = DEFAULT_PORT
    startup_timeout_sec: float = 10.0

    # Real-time scheduling for the UDP receiver subprocess.  Same slot
    # GMR would take on the sibling onboard deploy, so the recv worker
    # stays off core 4 (reserved for loco) and is preempt-able above
    # normal user tasks.
    net_recv_rt_pin: tuple[int, int] | None = (2, 40)
    """Pin UDP recv subprocess to (cpu_id, SCHED_FIFO priority).  Set to
    None to disable when running without CAP_SYS_NICE."""

    # BrainCo dex-hand
    enable_brainco_hand: bool = True
    """Initialize BraincoController and drive it from network/offline hand cmd."""
    brainco_hand_fps: int = 100
    brainco_hand_smooth_alpha: float = 0.45
    brainco_hand_scale: float = 1.0
    rest_hand_in_walk: bool = True
    """Send the rest pose to the hands whenever the body is in walk mode."""


def main(args: OnboardWoGmrBraincoArgs):
    curses.wrapper(lambda stdscr: _run_onboard(stdscr, args))


if __name__ == "__main__":
    main(tyro.cli(OnboardWoGmrBraincoArgs))
