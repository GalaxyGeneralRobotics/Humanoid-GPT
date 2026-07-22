"""Online motion-capture retarget subprocess.

Launches a background process that reads from PNLink or Xsens MVN,
runs GMR (General Motion Retargeting), and writes the latest qpos_full
into shared memory for the main deploy loop to consume.  Source
selection is controlled by the ``mocap_type`` string (``"pnlink"`` or
``"xsens"``; anything other than ``"xsens"`` falls back to PNLink).
"""

from __future__ import annotations

import time
import atexit
import threading
import numpy as np
import multiprocessing as mp
from collections import deque
from multiprocessing.sharedctypes import SynchronizedArray


# Hand detection joint names (for PNLink)
_HAND_JOINTS = {
    "left":  {"wrist": "LeftHand"},
    "right": {"wrist": "RightHand"},
}
_HAND_THRESHOLD = 0.05

# Keep IPC primitives/process handles alive for spawn context.
# If these objects are garbage-collected too early, spawned children may fail
# rebuilding SemLock with FileNotFoundError.
_RETARGET_SESSIONS: list[dict] = []


def _detect_hand_open(frame, wrist: str, threshold: float = _HAND_THRESHOLD):
    """Detect hand open/close from finger-to-thumb distance."""
    try:
        if wrist == "RightHand":
            dist = np.linalg.norm(np.array(frame["RightHandIndex3"][0]) - np.array(frame["RightHandThumb3"][0]))
        else:
            dist = np.linalg.norm(np.array(frame["LeftHandIndex3"][0]) - np.array(frame["LeftHandThumb3"][0]))
        return dist > threshold, dist
    except KeyError:
        return False, 0.0


def _retarget_worker(
    buf, buf_hand, ts, ready_evt, stop_evt,
    robot, actual_human_height, mocap_type,
    buffer_ms, rt_pin,
    xsens_host="0.0.0.0", xsens_port=9763, xsens_protocol="tcp",
):
    """Worker process: mocap -> GMR retarget -> shared memory.

    ``rt_pin``: optional ``(cpu_id, fifo_priority)`` tuple.  When set, pin
    this process to ``cpu_id`` and run it under ``SCHED_FIFO`` at
    ``fifo_priority``.  Intended for resource-constrained on-board targets
    (e.g. Jetson via ``deploy.onboard_deploy.play_track_onboard``) where
    isolating GMR on a dedicated core measurably reduces mocap jitter.
    On general-purpose workstations leave this ``None`` — pinning a single
    core there would only contend with viewer / camera / IDE threads.
    """
    if rt_pin is not None:
        import os
        cpu_id, fifo_prio = rt_pin
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(fifo_prio)))
        except (OSError, PermissionError):
            pass

    from gmr import GeneralMotionRetargeting as GMR

    if (mocap_type or "").lower() == "xsens":
        from deploy.xsens.client import XsensClient
        client = XsensClient(
            host=xsens_host, port=xsens_port, protocol=xsens_protocol,
        )
        client.start_thread()
        get_frame = lambda: client.get_frame_data(timeout=0.5)
        src_human = "fbx_xsens"
    else:
        from noitom import NoitomClient
        client = NoitomClient()
        client.start_thread()
        get_frame = lambda: client.get_frame_data(timeout=True)
        src_human = "fbx_noitom"

    retarget = GMR(src_human=src_human, tgt_robot=robot, actual_human_height=actual_human_height)

    qpos_last = None
    ema_alpha = 0.75

    # -- Jitter buffer: absorb Noitom delivery timing jitter --
    _use_jbuf = buffer_ms > 0
    if _use_jbuf:
        _nominal_hz = 90.0
        _target_depth = max(1, round(buffer_ms / 1000.0 * _nominal_hz))
        _jbuf: deque[tuple[np.ndarray, np.ndarray]] = deque()
        _jbuf_lock = threading.Lock()
        _jbuf_filled = threading.Event()

        def _jitter_output():
            dt_out = 1.0 / _nominal_hz
            _out_qpos = None
            _out_hand = np.zeros(4, dtype=np.float32)
            _jbuf_filled.wait()
            if not ready_evt.is_set():
                ready_evt.set()
            while not stop_evt.is_set():
                popped = False
                with _jbuf_lock:
                    if _jbuf:
                        _out_qpos, _out_hand = _jbuf.popleft()
                        depth = len(_jbuf)
                        popped = True
                    else:
                        depth = 0
                if popped and _out_qpos is not None:
                    with buf_hand.get_lock():
                        np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = _out_hand
                    with buf.get_lock(), ts.get_lock():
                        np.frombuffer(buf.get_obj(), dtype=np.float32, count=_out_qpos.size)[:] = _out_qpos
                        ts.value = time.time()
                depth_err = depth - _target_depth
                dt_out = (1.0 / _nominal_hz) * (1.0 - 0.02 * depth_err)
                dt_out = max(0.005, min(0.030, dt_out))
                time.sleep(dt_out)

        threading.Thread(target=_jitter_output, daemon=True).start()
        print(f"[Retarget] Jitter buffer enabled: {buffer_ms:.0f} ms ({_target_depth} frames)")

    try:
        while not stop_evt.is_set():
            frame = get_frame()
            if frame is None:
                continue

            # Hand detection
            l_open, l_dist = _detect_hand_open(frame, **_HAND_JOINTS["left"])
            r_open, r_dist = _detect_hand_open(frame, **_HAND_JOINTS["right"])
            hand_data = np.array([float(l_open), l_dist, float(r_open), r_dist], dtype=np.float32)
            if not _use_jbuf:
                with buf_hand.get_lock():
                    np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand_data

            # Retarget
            try:
                qpos = retarget.retarget(frame)
            except Exception as e:
                import traceback
                print(f"[Retarget] error: {e}\n{traceback.format_exc()}")
                continue

            # EMA smoothing
            if qpos_last is not None:
                qpos = qpos_last * ema_alpha + qpos * (1.0 - ema_alpha)
            qpos_last = qpos.copy()
            qpos = np.asarray(qpos, dtype=np.float32)

            if _use_jbuf:
                with _jbuf_lock:
                    _jbuf.append((qpos.copy(), hand_data))
                    if not _jbuf_filled.is_set() and len(_jbuf) >= _target_depth:
                        _jbuf_filled.set()
                    while len(_jbuf) > _target_depth * 3:
                        _jbuf.popleft()
            else:
                with buf.get_lock(), ts.get_lock():
                    mv = np.frombuffer(buf.get_obj(), dtype=np.float32, count=qpos.size)
                    mv[:] = qpos
                    ts.value = time.time()

            if not _use_jbuf and not ready_evt.is_set():
                ready_evt.set()
    finally:
        if hasattr(client, "stop"):
            client.stop()


def _visualize_worker(buf, stop_evt, robot="unitree_g1"):
    from gmr import RobotMotionViewer
    viewer = RobotMotionViewer(robot_type=robot, motion_fps=120.0)
    while not stop_evt.is_set():
        with buf.get_lock():
            qpos = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        viewer.step(root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:])


def start_realtime_retarget(
    robot: str = "unitree_g1",
    dof_full: int = 36,
    actual_human_height: float = 1.6,
    visualize_retarget: bool = False,
    mocap_type: str = "pnlink",
    buffer_ms: float = 0.0,
    rt_pin: tuple[int, int] | None = None,
    xsens_host: str = "0.0.0.0",
    xsens_port: int = 9763,
    xsens_protocol: str = "tcp",
) -> tuple[SynchronizedArray, ...]:
    """Launch retarget worker and return shared buffers.

    Args:
        rt_pin: Optional ``(cpu_id, fifo_priority)`` for the GMR subprocess.
            When set, the worker pins itself to ``cpu_id`` and runs under
            ``SCHED_FIFO`` at ``fifo_priority``.  Use only on resource-
            constrained on-board targets (e.g. Jetson); leave ``None`` for
            workstation runs (``deploy/play_track.py``, ``collect_data``,
            etc.) to avoid contending with viewer / camera / IDE threads.

    Returns:
        (buf_qpos, ts, buf_hand)
        - buf_qpos: Array('f', dof_full) – latest retargeted full qpos
        - ts: Value('d') – timestamp of last update
        - buf_hand: Array('f', 4) – [left_open, left_dist, right_open, right_dist]
    """
    # Use "spawn" to avoid inheriting X11/GL state from parent process.
    # This prevents xcb thread-sequence crashes when multiple GUI viewers run.
    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", dof_full, lock=True)
    buf_hand = ctx.Array("f", 4, lock=True)
    ts = ctx.Value("d", 0.0)
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()

    p = ctx.Process(
        target=_retarget_worker,
        args=(buf, buf_hand, ts, ready_evt, stop_evt,
              robot, actual_human_height, mocap_type, buffer_ms, rt_pin,
              xsens_host, xsens_port, xsens_protocol),
        daemon=True,
    )
    p.start()

    vis_p = None
    if visualize_retarget:
        vis_p = ctx.Process(target=_visualize_worker, args=(buf, stop_evt), daemon=True)
        vis_p.start()
        atexit.register(lambda: vis_p.terminate())

    # Persist references so spawn children can always rebuild synchronization
    # primitives from valid OS handles.
    _RETARGET_SESSIONS.append({
        "proc": p,
        "vis_proc": vis_p,
        "ready_evt": ready_evt,
        "stop_evt": stop_evt,
        "buf": buf,
        "buf_hand": buf_hand,
        "ts": ts,
    })

    return buf, ts, buf_hand


def read_mocap_buffer(buf, ts) -> tuple[np.ndarray, float]:
    """Read the latest qpos_full and timestamp from shared memory."""
    with buf.get_lock(), ts.get_lock():
        qpos_full = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        timestamp = ts.value
    if np.all(qpos_full == 0):
        qpos_full[3] = 1.0
    return qpos_full, timestamp


def read_hand_buffer(buf_hand) -> tuple[bool, float, bool, float] | None:
    """Read hand open/close state from shared memory."""
    if buf_hand is None:
        return None
    with buf_hand.get_lock():
        data = np.frombuffer(buf_hand.get_obj(), dtype=np.float32).copy()
    return bool(data[0]), data[1], bool(data[2]), data[3]
