"""Benchmark the FULL online pipeline with GMR in a separate subprocess.

Mirrors the real production architecture in play_track_onboard (mode 1):
GMR retarget runs in an independent subprocess writing to shared memory,
while the main loop reads from shared memory and runs the tracking policy.

Uses synthetic mocap frames (random perturbations of a T-pose) so no
real Noitom connection is needed.

Segments measured (main loop):

    00_get_sensor_state  DDS attribute read (29 motors + IMU)
    01_read_mocap_buf    shared memory read (qpos_full from GMR subprocess)
    02_live_convert      LiveRefConverter: MuJoCo FK + ref dict
    03_infer_total       tracking policy ONNX inference
    04_low_cmd_pack      motor command packing
    05_crc               CRC computation
    99_step_total        whole step

GMR subprocess throughput is reported separately.

Usage:
    python -m deploy.onboard_deploy.bench_online_full --net eth0
    python -m deploy.onboard_deploy.bench_online_full --onnx-track model.onnx
"""

from __future__ import annotations

from deploy.onboard_deploy._bench_utils import aarch64_preload

aarch64_preload()

import sys


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

import time
import gc
import os
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import tyro

from deploy.onboard_deploy._bench_utils import (
    BenchProfiler,
    print_budget_summary,
    print_environment_info,
)
from tracking.policy import Args as PolicyArgs
from tracking.policy import get_policy_onnx
from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from deploy.onboard_deploy.play_track_onboard import LiveRefConverter


@dataclass
class BenchArgs:
    """Full-pipeline benchmark with GMR subprocess on a connected G1."""

    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    policy_type: str = "mlp"
    use_trt: bool = True
    num_warmup: int = 50
    num_iters: int = 500
    freq: int = 50
    net: str = "eth0"
    convert_xml_path: str = str(consts.TRACK_XML)
    human_height: float = 1.7
    mocap_type: str = "pnlink"
    fast_path: bool = True


# ---------------------------------------------------------------------------
# GMR subprocess
# ---------------------------------------------------------------------------

def _random_quat(rng: np.random.Generator, max_angle_deg: float = 15.0):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = rng.uniform(0, np.radians(max_angle_deg))
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s],
                    dtype=np.float64)


def _make_synthetic_frame(body_names: list[str], rng: np.random.Generator) -> dict:
    frame = {}
    for name in body_names:
        pos = np.array([0.0, 0.0, 0.9]) + rng.uniform(-0.05, 0.05, 3)
        quat = _random_quat(rng)
        frame[name] = [pos, quat]
    return frame


def _gmr_worker(
    buf: mp.Array,
    ts: mp.Value,
    counter: mp.Value,
    ready_evt: mp.Event,
    stop_evt: mp.Event,
    human_height: float,
    mocap_type: str,
):
    """Subprocess: generate synthetic mocap → GMR retarget → shared memory."""
    import sys as _sys

    class _WarpFilter:
        def __init__(self, stream):
            self._stream = stream
        def write(self, msg):
            if "Failed to import warp" not in msg and "Failed to import mujoco_warp" not in msg:
                self._stream.write(msg)
        def flush(self):
            self._stream.flush()
        def __getattr__(self, name):
            return getattr(self._stream, name)

    _sys.stdout = _WarpFilter(_sys.stdout)
    from gmr import GeneralMotionRetargeting as GMR
    _sys.stdout = _sys.stdout._stream

    # Pin GMR subprocess to core 2 with SCHED_FIFO
    try:
        os.sched_setaffinity(0, {2})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(40))
    except (OSError, PermissionError):
        pass

    src_human = "fbx_xsens" if mocap_type == "xsens" else "fbx_noitom"
    gmr = GMR(src_human=src_human, tgt_robot="unitree_g1",
              actual_human_height=human_height)
    body_names = list(gmr.human_scale_table.keys())
    rng = np.random.default_rng(123)

    buf_size = len(buf)
    ready_evt.set()

    while not stop_evt.is_set():
        frame = _make_synthetic_frame(body_names, rng)
        qpos = gmr.retarget(frame)
        qpos = np.asarray(qpos, dtype=np.float32)[:buf_size]

        with buf.get_lock(), ts.get_lock():
            np.frombuffer(buf.get_obj(), dtype=np.float32)[:] = qpos
            ts.value = time.time()
        with counter.get_lock():
            counter.value += 1


def _read_mocap_buffer(buf, ts) -> tuple[np.ndarray, float]:
    """Read latest qpos_full from shared memory (same as deploy/retarget.py)."""
    with buf.get_lock(), ts.get_lock():
        qpos_full = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        timestamp = ts.value
    if np.all(qpos_full == 0):
        qpos_full[3] = 1.0
    return qpos_full, timestamp


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _step_full(
    prof: BenchProfiler,
    low_ctrl,
    buf,
    ts,
    live_converter: LiveRefConverter,
    infer_fn: G1TrackInferFn,
    prev_ref: dict | None,
) -> dict:
    step_start = time.perf_counter()

    with prof.time("00_get_sensor_state"):
        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()

    with prof.time("01_read_mocap_buf"):
        qpos_full, _ = _read_mocap_buffer(buf, ts)
        qpos_full = np.asarray(qpos_full, dtype=np.float32)

    with prof.time("02_live_convert"):
        ref_new = live_converter.convert(qpos_full)

    ref_state = {"ref_curr": prev_ref if prev_ref is not None else ref_new,
                 "ref_next": ref_new}

    with prof.time("03_infer_total"):
        motor_targets = infer_fn.infer_onnx_real_fast(
            root_quat, root_gyro, jnt_qpos, jnt_qvel, ref_state
        )

    flat_targets = np.asarray(motor_targets).flatten()

    with prof.time("04_low_cmd_pack"):
        low_ctrl.fast_pack_motor_cmd(flat_targets)

    with prof.time("05_crc"):
        low_ctrl.fast_compute_crc()

    prof.record("99_step_total", (time.perf_counter() - step_start) * 1e3)
    return ref_new


def main(args: BenchArgs) -> None:
    print_environment_info(extra_packages=["gmr"])

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from deploy.real_robot import LowLevelControlG1

    ctrl_dt = 1.0 / args.freq
    dof_full = 36  # matches deploy/retarget.py default

    # DDS
    print(f"\nDDS init on '{args.net}' ...")
    ChannelFactoryInitialize(0, args.net)
    print("Connecting to robot (DDS publish: disabled) ...")
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=True)
    print("Robot connected.")

    # Launch GMR subprocess
    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", dof_full, lock=True)
    ts = ctx.Value("d", 0.0)
    counter = ctx.Value("i", 0)
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()

    print(f"Spawning GMR subprocess (height={args.human_height}, mocap={args.mocap_type}) ...")
    gmr_proc = ctx.Process(
        target=_gmr_worker,
        args=(buf, ts, counter, ready_evt, stop_evt,
              args.human_height, args.mocap_type),
        daemon=True,
    )
    gmr_proc.start()

    print("Waiting for GMR subprocess ready ...")
    ready_evt.wait(timeout=30)
    if not ready_evt.is_set():
        print("ERROR: GMR subprocess did not become ready in 30s")
        gmr_proc.terminate()
        return
    # Let GMR run a few iterations to fill buffer
    time.sleep(0.5)
    print(f"  GMR subprocess ready (pid={gmr_proc.pid})")

    # Policy + MuJoCo
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    print(f"Loading policy: {args.onnx_track}")
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track, policy_type=args.policy_type
    )
    track_policy = get_policy_onnx(
        policy_args, use_trt=args.use_trt, strict_trt=False
    )

    print(f"Loading MuJoCo model: {args.convert_xml_path}")
    phantom_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    phantom_model.opt.timestep = 0.001
    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # Pin main process to core 4 with SCHED_FIFO real-time priority
    try:
        os.sched_setaffinity(0, {4})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        print("  Main loop: core 4, SCHED_FIFO priority 50")
    except (OSError, PermissionError) as e:
        print(f"  Main loop: core pinning/RT failed ({e}), running default scheduler")

    # Warmup
    print(f"\nWarmup ({args.num_warmup} iters)...")
    warmup_prof = BenchProfiler()
    prev_ref = None
    next_t = time.perf_counter()
    for _ in range(args.num_warmup):
        next_t += ctrl_dt
        prev_ref = _step_full(
            warmup_prof, low_ctrl, buf, ts, live_converter, infer_fn, prev_ref
        )
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Benchmark
    gc.collect()
    gc.disable()
    prof = BenchProfiler()
    prev_ref = None
    live_converter.reset()
    with counter.get_lock():
        counter.value = 0
    t_bench_start = time.perf_counter()

    print(f"Benchmarking ({args.num_iters} iters)...")
    next_t = time.perf_counter()
    for _ in range(args.num_iters):
        next_t += ctrl_dt
        prev_ref = _step_full(
            prof, low_ctrl, buf, ts, live_converter, infer_fn, prev_ref
        )
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    t_bench_end = time.perf_counter()
    gc.enable()

    # Stop GMR subprocess
    stop_evt.set()
    gmr_proc.join(timeout=5)
    if gmr_proc.is_alive():
        gmr_proc.terminate()

    # GMR subprocess stats
    with counter.get_lock():
        gmr_iters = counter.value
    bench_duration = t_bench_end - t_bench_start
    gmr_hz = gmr_iters / bench_duration if bench_duration > 0 else 0

    header = (
        f"ONLINE FULL pipeline (GMR subprocess)  |  net={args.net}  "
        f"TRT={args.use_trt}  freq={args.freq}Hz  "
        f"mocap={args.mocap_type}  height={args.human_height}"
    )
    print(prof.summary(header, iters=args.num_iters))
    print(f"\n  GMR subprocess: {gmr_iters} retargets in {bench_duration:.1f}s "
          f"= {gmr_hz:.1f} Hz")
    print_budget_summary(
        np.asarray(prof.timings["99_step_total"]), target_freq_hz=args.freq
    )

    try:
        low_ctrl.set_motor_damping()
    except Exception:
        pass


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
