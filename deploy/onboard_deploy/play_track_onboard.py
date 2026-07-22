"""Onboard deployment for Unitree G1 — runs directly on the robot via SSH.

Replaces the pygame-based GUI with a curses terminal interface so that
the entire Humanoid-GPT tracking pipeline (walk / online retarget / offline
tracking) can run over an SSH session on the G1's onboard Jetson computer.

Usage (after SSH into G1):
    python -m deploy.onboard_deploy.play_track_onboard
    python -m deploy.onboard_deploy.play_track_onboard --no-mocap

Modes (number keys):
    0 = Walk (velocity commands via WASD/QE)
    1 = Online retarget (Noitom PNLink over WiFi)
    2+ = Offline trajectory tracking (from --track-dir)
"""

from __future__ import annotations

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
import mujoco
sys.stdout = sys.stdout._stream
del _MujocoWarpFilter
import curses
import select
import logging
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass


from tracking import constants as consts
from tracking.constants import KPT_NAMES
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config, apply_ema_qpos

from deploy.constants import (
    DEFAULT_QPOS as DEFAULT_QPOS_JOINT,
    KEYBOARD_MAX_SPEED,
    KPs_walking,
    KDs_walking,
)
from deploy.walk_policy import WalkPolicy


# ---------------------------------------------------------------------------
# Logging (curses owns stdout, so log to file)
# ---------------------------------------------------------------------------

_LOG_DIR = Path("/tmp/humanoid_gpt_onboard")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"deploy_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=str(_LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
_log = logging.getLogger("onboard")


# ---------------------------------------------------------------------------
# Lightweight helpers (avoid importing deploy.play_track which pulls pygame)
# ---------------------------------------------------------------------------


def _damp(v: float, rate: float) -> float:
    if v > rate:
        return v - rate
    if v < -rate:
        return v + rate
    return 0.0


def _tree_index(tree: dict, idx: int) -> dict:
    return {
        k: (v[idx][None] if isinstance(v, np.ndarray) else v)
        for k, v in tree.items()
    }


def _make_bar(val: float, vmax: float, width: int = 20) -> str:
    mid = width // 2
    bar = list("." * width)
    bar[mid] = "|"
    filled = int(abs(val) / max(vmax, 1e-6) * mid)
    if val > 0.001:
        for i in range(mid + 1, min(mid + 1 + filled, width)):
            bar[i] = "#"
    elif val < -0.001:
        for i in range(max(mid - filled, 0), mid):
            bar[i] = "#"
    return "".join(bar)


@dataclass
class HighCommand:
    vel_lin_x: float = 0.0
    vel_lin_y: float = 0.0
    vel_ang_yaw: float = 0.0
    mode: int = 0
    kill: bool = False


class MocapBuffer:
    def __init__(self, buf, ts):
        self._buf, self._ts = buf, ts

    def read(self):
        from deploy.retarget import read_mocap_buffer

        return read_mocap_buffer(self._buf, self._ts)


def check_mocap_health(qpos_full: np.ndarray) -> str:
    """Lightweight sanity check on retargeted qpos (no FK needed)."""
    h = qpos_full[2]
    jnt_err = float(np.mean(np.abs(qpos_full[7:] - consts.DEFAULT_QPOS[7:])))
    h_ok = 0.3 < h < 1.2
    j_ok = jnt_err < 0.5
    tag = "OK" if (h_ok and j_ok) else "WARN"
    return f"{tag}  h={h:.2f}  jnt_err={jnt_err:.2f}"


# ---------------------------------------------------------------------------
# LiveRefConverter (duplicated from play_track.py to avoid the pygame chain)
# ---------------------------------------------------------------------------


def _batch_pose_delta_to_twist_inplace(T_prev, T_curr, inv_dt, out, R_delta, skew):
    """Compute twist in-place with pre-allocated scratch buffers."""
    nk = T_prev.shape[0]
    R0 = T_prev[:, :3, :3]
    R1 = T_curr[:, :3, :3]

    # linear velocity (world frame)
    out[:, 3:] = (T_curr[:, :3, 3] - T_prev[:, :3, 3]) * inv_dt

    # R_delta = R0^T @ R1
    np.einsum("kij,kjl->kil", R0.transpose(0, 2, 1), R1, out=R_delta)

    # SO3 log -> rotvec (in-place into skew buffer)
    tr = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]
    cos_theta = (tr - 1.0) * 0.5
    np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
    theta = np.arccos(cos_theta)

    skew[:, 0] = R_delta[:, 2, 1] - R_delta[:, 1, 2]
    skew[:, 1] = R_delta[:, 0, 2] - R_delta[:, 2, 0]
    skew[:, 2] = R_delta[:, 1, 0] - R_delta[:, 0, 1]

    small = theta < 1e-6
    safe_theta = np.where(small, 1.0, theta)
    k = np.where(small, 0.5, safe_theta / (2.0 * np.sin(safe_theta)))
    skew *= k[:, None]

    # w_w = R0 @ rotvec * inv_dt
    np.einsum("kij,kj->ki", R0, skew, out=out[:, :3])
    out[:, :3] *= inv_dt


def _quat_to_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _wrap_to_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LiveRefConverter:
    """Convert raw mocap qpos_full to ref_state dict via MuJoCo FK.

    Keypoint velocities are computed via finite differences.  For real-robot
    deployment, ``set_robot_initial_pose()`` rebiases the reference root to
    align with the robot's IMU yaw at the moment tracking begins.
    """

    def __init__(self, mj_model: mujoco.MjModel, ctrl_dt: float):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.ctrl_dt = ctrl_dt
        self.inv_dt = 1.0 / ctrl_dt
        self.kpt_body_ids = np.array([mj_model.body(n).id for n in KPT_NAMES])
        self._prev_kpt = None
        self._prev_gv = np.eye(4, dtype=np.float32)
        self._prev_qpos = None
        self._has_prev = False
        self._ref_xy = self._ref_yaw = None
        self._rob_xy = self._rob_yaw = None

        nk = len(self.kpt_body_ids)
        self._nk = nk
        self._kpt2wrd_a = np.zeros((nk, 4, 4), dtype=np.float32)
        self._kpt2wrd_a[:, 3, 3] = 1.0
        self._kpt2wrd_b = np.zeros((nk, 4, 4), dtype=np.float32)
        self._kpt2wrd_b[:, 3, 3] = 1.0
        self._kpt_cur = 0  # 0 = using _a as current, 1 = using _b
        self._kpt2gv = np.zeros((nk, 4, 4), dtype=np.float32)
        self._gvi2wrd_pose = np.eye(4, dtype=np.float32)
        self._gv_inv = np.eye(4, dtype=np.float32)
        self._cvel_wrd = np.zeros((nk, 6), dtype=np.float32)
        self._cvel_gv = np.zeros((nk, 6), dtype=np.float32)
        self._R_delta = np.zeros((nk, 3, 3), dtype=np.float32)
        self._skew = np.zeros((nk, 3), dtype=np.float32)
        self._gv_vel = np.zeros(3, dtype=np.float32)
        self._qvel = np.zeros(35, dtype=np.float32)

    def reset(self):
        self._prev_qpos = None
        self._has_prev = False
        self._kpt_cur = 0
        self._ref_xy = self._ref_yaw = None
        self._rob_xy = self._rob_yaw = None

    def set_robot_initial_pose(self, robot_quat, robot_xy):
        self._rob_yaw = _quat_to_yaw(robot_quat)
        self._rob_xy = robot_xy[:2].copy()

    def _rebias_qpos(self, qpos_full):
        if self._rob_yaw is None:
            return qpos_full
        qpos = qpos_full.copy()
        if self._ref_xy is None:
            self._ref_yaw = _quat_to_yaw(qpos[3:7])
            self._ref_xy = qpos[:2].copy()
        ref_yaw = _quat_to_yaw(qpos[3:7])
        d_yaw = _wrap_to_pi(ref_yaw - self._ref_yaw)
        d_xy = qpos[:2] - self._ref_xy
        yaw_off = _wrap_to_pi(self._rob_yaw - self._ref_yaw)
        c, s = np.cos(yaw_off), np.sin(yaw_off)
        rot_dxy = np.array(
            [c * d_xy[0] - s * d_xy[1], s * d_xy[0] + c * d_xy[1]], dtype=np.float32
        )
        qpos[:2] = self._rob_xy + rot_dxy
        new_yaw = self._rob_yaw + d_yaw
        d_ya = _wrap_to_pi(new_yaw - ref_yaw)
        c2, s2 = np.cos(d_ya / 2), np.sin(d_ya / 2)
        q_dz = np.array([c2, 0.0, 0.0, s2], dtype=np.float32)
        q_new = _quat_mul_wxyz(q_dz, qpos[3:7])
        qpos[3:7] = q_new / np.clip(np.linalg.norm(q_new), 1e-8, None)
        return qpos

    def convert(self, qpos_full: np.ndarray) -> dict:
        qpos_full = self._rebias_qpos(qpos_full)
        self.mj_data.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

        # gvi2wrd: navigation frame (yaw-only rotation + xy position)
        q = qpos_full[3:7]
        w, x, y, z = q[0], q[1], q[2], q[3]
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z
        R00 = 1.0 - 2.0 * (yy + zz)
        R10 = 2.0 * (xy + wz)
        # project x-axis onto xy-plane for navigation frame
        norm_xy = np.sqrt(R00 * R00 + R10 * R10)
        inv_norm = 1.0 / (norm_xy + 1e-12)
        cx = R00 * inv_norm
        sx = R10 * inv_norm

        gv = self._gvi2wrd_pose
        gv[0, 0] = cx;  gv[0, 1] = -sx; gv[0, 2] = 0.0
        gv[1, 0] = sx;  gv[1, 1] = cx;  gv[1, 2] = 0.0
        gv[2, 0] = 0.0; gv[2, 1] = 0.0; gv[2, 2] = 1.0
        gv[0, 3] = qpos_full[0]
        gv[1, 3] = qpos_full[1]
        gv[2, 3] = 0.0

        # SE(3) analytic inverse: R^T, -R^T @ t
        gv_inv = self._gv_inv
        gv_inv[0, 0] = cx;  gv_inv[0, 1] = sx;  gv_inv[0, 2] = 0.0
        gv_inv[1, 0] = -sx; gv_inv[1, 1] = cx;  gv_inv[1, 2] = 0.0
        gv_inv[2, 0] = 0.0; gv_inv[2, 1] = 0.0; gv_inv[2, 2] = 1.0
        gv_inv[0, 3] = -(cx * gv[0, 3] + sx * gv[1, 3])
        gv_inv[1, 3] = -(-sx * gv[0, 3] + cx * gv[1, 3])
        gv_inv[2, 3] = 0.0

        # keypoint poses from MuJoCo FK (double-buffer: no copy needed)
        kpt2wrd = self._kpt2wrd_a if self._kpt_cur == 0 else self._kpt2wrd_b
        kpt_prev = self._kpt2wrd_b if self._kpt_cur == 0 else self._kpt2wrd_a
        kpt2wrd[:, :3, 3] = self.mj_data.xpos[self.kpt_body_ids]
        kpt2wrd[:, :3, :3] = self.mj_data.xmat[self.kpt_body_ids].reshape(-1, 3, 3)

        # kpt2gv = gv_inv @ kpt2wrd (batch matmul via einsum)
        np.einsum("ij,kjl->kil", gv_inv, kpt2wrd, out=self._kpt2gv)

        # Keypoint velocities via finite differences
        cvel_gv = self._cvel_gv
        if self._has_prev:
            _batch_pose_delta_to_twist_inplace(
                kpt_prev, kpt2wrd, self.inv_dt,
                self._cvel_wrd, self._R_delta, self._skew
            )
            cvel_gv[:, :3] = self._cvel_wrd[:, :3] @ gv[:3, :3]
            cvel_gv[:, 3:] = self._cvel_wrd[:, 3:] @ gv[:3, :3]
        else:
            cvel_gv[:] = 0.0
        self._kpt_cur ^= 1  # swap buffers

        # gv velocity
        gv_vel = self._gv_vel
        if self._has_prev:
            # c2p = prev_inv @ gv  (2D rotation: just angle diff + translation)
            pc = self._prev_gv[0, 0]; ps = self._prev_gv[1, 0]
            # prev_inv rotation: [pc, ps; -ps, pc]
            dx = gv[0, 3] - self._prev_gv[0, 3]
            dy = gv[1, 3] - self._prev_gv[1, 3]
            gv_vel[0] = (pc * dx + ps * dy) * self.inv_dt
            gv_vel[1] = (-ps * dx + pc * dy) * self.inv_dt
            # yaw rate
            gv_vel[2] = np.arctan2(
                pc * sx - ps * cx, pc * cx + ps * sx
            ) * self.inv_dt
        else:
            gv_vel[:] = 0.0
        self._prev_gv[:] = gv
        self._has_prev = True

        # Joint velocity via finite differences
        qvel = self._qvel
        if self._prev_qpos is not None:
            qvel[6:] = (qpos_full[7:] - self._prev_qpos[7:]) * self.inv_dt
        else:
            qvel[:] = 0.0
        self._prev_qpos = qpos_full.copy()

        return {
            "qpos": qpos_full[None].astype(np.float32),
            "qvel": qvel[None, :].astype(np.float32),
            "kpt2gv_pose": self._kpt2gv[None].astype(np.float32),
            "kpt_cvel_in_gv": cvel_gv[None].astype(np.float32),
            "gv2wrd_pose": gv[None].astype(np.float32),
            "gv_vel": gv_vel[None].astype(np.float32),
        }


def load_offline_motions(track_dir: str, mj_model, freq: int = 50) -> list[dict]:
    folder = Path(track_dir)
    files = [folder] if folder.is_file() else sorted(folder.glob("*.npz"))
    motions: list[dict] = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            continue
        data["qpos"] = apply_ema_qpos(data["qpos"])
        freq_src = float(data.get("frequency", 50))
        kpt = qpos2kpt(
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
        motions.append({"data": kpt, "filename": f.name})
    return motions


# ---------------------------------------------------------------------------
# Terminal command interface (curses)
# ---------------------------------------------------------------------------

_MODE_NAMES = ["Walk", "Online"]


class TerminalCMD:
    """Thread-safe keyboard command interface with curses display."""

    DELTA = 0.1
    VMAX = KEYBOARD_MAX_SPEED
    HOLD_SEC = 0.5  # velocity persists for this long after last keypress

    def __init__(self, num_track_ref: int = 0):
        self.num_modes = 2 + num_track_ref
        self._lock = threading.Lock()
        self._mode = 0
        self._vx = self._vy = self._vyaw = 0.0
        self._kill = False
        self._estop = False
        self._reset_req = False
        self._robot_status = "Initializing"
        self._mocap_status = "N/A"
        self._freq_hz = 0.0
        self._total_steps = 0
        self._track_info = ""
        self._mocap_health = ""
        self._last_key_time = {"x": 0.0, "y": 0.0, "w": 0.0}

    # -- UI thread -----------------------------------------------------------

    def poll_key(self, stdscr) -> str | None:
        """Drain keyboard buffer via direct stdin read (SSH-compatible)."""
        axes_active: set[str] = set()
        special: str | None = None
        fd = sys.stdin.fileno()
        while select.select([fd], [], [], 0)[0]:
            try:
                ch = os.read(fd, 1)
            except OSError:
                break
            if not ch:
                break
            _log.debug("key received: %d (%r)", ch[0], chr(ch[0]))
            with self._lock:
                s = self._handle_key(ch[0], axes_active)
                if s is not None:
                    special = s

        now = time.monotonic()
        with self._lock:
            for axis in axes_active:
                self._last_key_time[axis] = now
            for axis, attr in [("x", "_vx"), ("y", "_vy"), ("w", "_vyaw")]:
                if axis not in axes_active and (now - self._last_key_time[axis]) > self.HOLD_SEC:
                    setattr(self, attr, 0.0)

        return special

    def _handle_key(self, key: int, axes: set) -> str | None:
        D, M = self.DELTA, self.VMAX
        if key in (ord("w"), ord("W")):
            self._vx = min(M, self._vx + D)
            axes.add("x")
        elif key in (ord("s"), ord("S")):
            self._vx = max(-M, self._vx - D)
            axes.add("x")
        elif key in (ord("a"), ord("A")):
            self._vy = min(M, self._vy + D)
            axes.add("y")
        elif key in (ord("d"), ord("D")):
            self._vy = max(-M, self._vy - D)
            axes.add("y")
        elif key in (ord("q"), ord("Q")):
            self._vyaw = min(M, self._vyaw + D)
            axes.add("w")
        elif key in (ord("e"), ord("E")):
            self._vyaw = max(-M, self._vyaw - D)
            axes.add("w")
        elif key in (ord("r"), ord("R")):
            self._reset_req = True
        elif key == ord(" "):
            self._estop = True
            self._kill = True
        elif key == 27:
            self._kill = True
        elif key in (10, 13):
            return "confirm"
        elif ord("0") <= key <= ord("9"):
            m = key - ord("0")
            if m < self.num_modes:
                old = self._mode
                self._mode = m
                _log.info("Mode switch: %d -> %d", old, m)
        if axes:
            _log.info("vel cmd: vx=%.2f vy=%.2f vyaw=%.2f", self._vx, self._vy, self._vyaw)
        return None

    def draw(self, stdscr):
        try:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            if h < 16 or w < 50:
                stdscr.addnstr(0, 0, "Terminal too small (min 50x16)", w)
                stdscr.refresh()
                return

            has_color = curses.has_colors()

            # Title
            title = " Humanoid-GPT Onboard Deploy "
            pad = max(0, w - len(title))
            attr_title = curses.A_REVERSE | curses.A_BOLD
            if has_color:
                attr_title = curses.color_pair(5) | curses.A_BOLD
            stdscr.addnstr(0, 0, " " * (pad // 2) + title + " " * (pad - pad // 2), w, attr_title)

            with self._lock:
                mode, vx, vy, vyaw = self._mode, self._vx, self._vy, self._vyaw
                rst, mst = self._robot_status, self._mocap_status
                fhz, steps = self._freq_hz, self._total_steps
                tinfo = self._track_info
                mhealth = self._mocap_health

            r = 2

            # Mode selector
            stdscr.addnstr(r, 2, "MODE", w - 2, curses.A_BOLD)
            c = 8
            for i in range(self.num_modes):
                name = _MODE_NAMES[i] if i < len(_MODE_NAMES) else f"Trk{i - 2}"
                label = f" {i}:{name} "
                attr = curses.A_REVERSE | curses.A_BOLD if i == mode else curses.A_DIM
                if c + len(label) < w:
                    stdscr.addnstr(r, c, label, w - c, attr)
                c += len(label) + 1
            r += 2

            # Velocity
            stdscr.addnstr(r, 2, "VELOCITY", w - 2, curses.A_BOLD)
            r += 1
            bw = min(20, w - 36)
            for lbl, val in [("X (W/S) ", vx), ("Y (A/D) ", vy), ("Yaw(Q/E)", vyaw)]:
                bar = _make_bar(val, self.VMAX, bw)
                col_attr = 0
                if has_color:
                    if val > 0.01:
                        col_attr = curses.color_pair(1)
                    elif val < -0.01:
                        col_attr = curses.color_pair(3)
                line = f"  {lbl} [{bar}] {val:+.2f}"
                stdscr.addnstr(r, 2, line, w - 2, col_attr)
                r += 1
            r += 1

            # Status
            stdscr.addnstr(r, 2, "STATUS", w - 2, curses.A_BOLD)
            r += 1
            rc = 0
            if has_color:
                if "Running" in rst:
                    rc = curses.color_pair(1)
                elif "Ready" in rst or "Waiting" in rst or "Connected" in rst:
                    rc = curses.color_pair(2)
                else:
                    rc = curses.color_pair(3)
            stdscr.addnstr(r, 4, "Robot: ", w - 4)
            stdscr.addnstr(r, 12, rst, w - 12, rc)
            r += 1
            mocap_line = mst
            if mhealth:
                mocap_line += f" | {mhealth}"
            mc = 0
            if has_color:
                if "WARN" in mocap_line:
                    mc = curses.color_pair(3)
                elif "Connected" in mocap_line:
                    mc = curses.color_pair(1)
            stdscr.addnstr(r, 4, "Mocap: ", w - 4)
            stdscr.addnstr(r, 12, mocap_line, w - 12, mc)
            r += 1
            stdscr.addnstr(r, 4, f"Freq: {fhz:5.1f} Hz   Steps: {steps}", w - 4)
            if tinfo:
                r += 1
                stdscr.addnstr(r, 4, f"Track: {tinfo}", w - 4)
            r += 2

            # Controls
            stdscr.addnstr(
                r, 2,
                "[0-9]Mode [WASDQE]Vel [R]Reset [Space]EStop [Esc]Quit",
                w - 2, curses.A_DIM,
            )
            r += 1
            stdscr.addnstr(r, 2, f"Log: {_LOG_FILE}", w - 2, curses.A_DIM)

            stdscr.refresh()
        except curses.error:
            pass

    # -- control thread ------------------------------------------------------

    def step_command(self) -> HighCommand:
        with self._lock:
            return HighCommand(
                vel_lin_x=self._vx,
                vel_lin_y=self._vy,
                vel_ang_yaw=self._vyaw,
                mode=self._mode,
                kill=self._kill,
            )

    def check_reset_request(self) -> bool:
        with self._lock:
            if self._reset_req:
                self._reset_req = False
                return True
            return False

    def set_status(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, f"_{k}", v)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Main onboard deployment loop
# ---------------------------------------------------------------------------


def _run_onboard(stdscr, args: "OnboardArgs"):
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
    from deploy.retarget import start_realtime_retarget, read_hand_buffer

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt = ctrl_dt)

    # Tracking policy
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
    )
    track_policy = get_policy_onnx(
        policy_args, use_trt=args.use_trt, strict_trt=False
    )
    walk_policy = WalkPolicy(args.onnx_walk)

    # Offline motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    _log.info("Loading offline reference motions...")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)
    for i, m in enumerate(ref_motions):
        _log.info(f"  Mode {i + 2}: {m['filename']} ({len(m['data']['qpos'])} frames)")

    keyboard = TerminalCMD(num_track_ref=len(ref_motions))

    # DDS channel + robot connection
    ChannelFactoryInitialize(0, args.net)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)
    keyboard.set_status(robot_status="Connected")
    _log.info(f"Robot connected via DDS on interface '{args.net}'")

    # Phantom MuJoCo model (reference FK only, no rendering)
    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = mujoco.MjModel.from_xml_path(xml_path)
    phantom_model.opt.timestep = 0.001

    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # Online retarget (Noitom PNLink over WiFi)
    mocap_buffer = None
    buf_hand = None
    if not args.no_mocap:
        try:
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=False,
                mocap_type=args.mocap_type,
                buffer_ms=args.buffer_ms,
                # Onboard-only: pin GMR to core 2 with SCHED_FIFO prio 40.
                # Mirrors bench_online_full.py and keeps mocap jitter low on
                # Jetson where CPU contention with the policy loop matters.
                rt_pin=args.gmr_rt_pin,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            keyboard.set_status(mocap_status=f"Connected ({args.mocap_type})")
            _log.info(f"Mocap retarget started ({args.mocap_type})")
        except Exception as e:
            keyboard.set_status(mocap_status=f"Failed: {e}")
            _log.error(f"Mocap init failed: {e}")

    # Hand controller
    hand_ctrl = None
    if args.enable_hand:
        try:
            hand_ctrl = Dex3Controller(net=args.net, re_init=False)
        except Exception as e:
            _log.error(f"Hand init failed: {e}")

    # -- shared mutable state for the control thread -------------------------
    last_mode = 0
    track_step = 0
    ref_traj = None
    last_left_hand = None
    last_right_hand = None
    prev_online_ref = None
    _freq = {"n": 0, "t0": time.time(), "total": 0}

    def locomotion_step():
        nonlocal last_mode, track_step, ref_traj
        nonlocal last_left_hand, last_right_hand, prev_online_ref

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
                    _log.info(f"Offline track start: {ref_motions[traj_idx]['filename']}")

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

            if mocap_buffer is not None:
                try:
                    qpos_full, _ = mocap_buffer.read()
                    health = check_mocap_health(qpos_full)
                    keyboard.set_status(mocap_health=health)
                except Exception:
                    keyboard.set_status(mocap_health="READ ERR")

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
                root_quat,
                root_gyro,
                jnt_qpos,
                jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(
                np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs
            )

            if buf_hand is not None:
                hand_cmd = read_hand_buffer(buf_hand)
                last_left_hand, last_right_hand = update_hand_from_mocap(
                    hand_ctrl, hand_cmd, last_left_hand, last_right_hand
                )

        last_mode = mode

        _freq["n"] += 1
        _freq["total"] += 1
        elapsed = time.time() - _freq["t0"]
        if elapsed >= 1.0:
            keyboard.set_status(freq_hz=_freq["n"] / elapsed, total_steps=_freq["total"])
            _freq["n"] = 0
            _freq["t0"] = time.time()

    # ---- Startup sequence --------------------------------------------------

    keyboard.set_status(robot_status="Damping — press [start] on remote")
    keyboard.draw(stdscr)

    while low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        keyboard.draw(stdscr)
        time.sleep(ctrl_dt)

    keyboard.set_status(robot_status="Standing up...")
    keyboard.draw(stdscr)
    low_ctrl.move_to_default_pos(duration=2.0)

    keyboard.set_status(robot_status="Ready — press [A] on remote")
    keyboard.draw(stdscr)

    while low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        keyboard.draw(stdscr)
        time.sleep(ctrl_dt)

    # ---- Main control loop -------------------------------------------------

    # Pin control thread to a dedicated core with real-time scheduling
    # to minimize jitter from CPU migration and kernel preemption.
    import os
    try:
        os.sched_setaffinity(0, {4})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        _log.info("RT scheduling: core 4, SCHED_FIFO priority 50")
    except (OSError, PermissionError) as e:
        _log.warning(f"RT scheduling unavailable ({e}), using default scheduler")

    import gc
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
                _log.info("Remote [select] — emergency stop")
                break
    except KeyboardInterrupt:
        _log.info("Ctrl+C received")
    finally:
        gc.enable()
        low_ctrl.set_motor_damping()
        keyboard.close()
        _log.info("Shutdown complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class OnboardArgs:
    """Onboard deployment on Unitree G1 (SSH terminal interface)."""

    # Policy paths
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    track_dir: str = "storage/test"
    convert_xml_path: str = str(consts.TRACK_XML)

    # Robot / control
    net: str = "eth0"
    """DDS network interface on G1 (internal motor bus, typically eth0)."""
    freq: int = 50
    debug: bool = False
    """Run without publishing motor commands (safe for testing)."""
    use_trt: bool = True
    """Use TensorRT for ONNX inference (recommended on Jetson)."""

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"
    """Mocap backend: 'pnlink' (Noitom over WiFi) or 'xsens'."""
    human_height: float = 1.7
    buffer_ms: float = 50.0

    # Real-time scheduling (on-board only)
    gmr_rt_pin: tuple[int, int] | None = (2, 40)
    """Pin GMR subprocess to (cpu_id, SCHED_FIFO priority).  Set to None to
    disable when running on a workstation or without CAP_SYS_NICE."""

    # Hand
    enable_hand: bool = False


def main(args: OnboardArgs):
    curses.wrapper(lambda stdscr: _run_onboard(stdscr, args))


if __name__ == "__main__":
    main(tyro.cli(OnboardArgs))
