"""Benchmark Noitom online stream updates used by onboard deploy mode 1.

This script mirrors only the mocap side of
``deploy.onboard_deploy.play_track_onboard``:

    Noitom PNLink -> GMR retarget subprocess -> shared qpos buffer -> read loop

It does not initialize DDS, load policies, touch robot motors, or run MuJoCo.
The reported update rate is measured from the shared-memory timestamp that the
retarget worker writes for the onboard control loop.

Usage:
    python -m deploy.onboard_deploy.bench_noitom_stream
    python -m deploy.onboard_deploy.bench_noitom_stream --duration-sec 60
    python -m deploy.onboard_deploy.bench_noitom_stream --buffer-ms 0
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tyro

import deploy.retarget as retarget_module
from deploy.onboard_deploy._bench_utils import aarch64_preload, print_environment_info
from deploy.retarget import read_mocap_buffer, start_realtime_retarget

aarch64_preload()


@dataclass
class BenchNoitomArgs:
    """Noitom online-stream benchmark matching onboard deploy's mocap path."""

    duration_sec: float = 30.0
    """Measured benchmark duration after warmup."""

    warmup_sec: float = 3.0
    """Warmup duration before statistics are recorded."""

    poll_hz: float = 200.0
    """Main-process polling rate for the shared mocap buffer."""

    human_height: float = 1.7
    """Human height passed to GMR retargeting."""

    buffer_ms: float = 50.0
    """Jitter-buffer latency in ms. 50 matches play_track_onboard default."""

    gmr_rt_pin: tuple[int, int] | None = (2, 40)
    """Pin GMR subprocess to (cpu_id, SCHED_FIFO priority), as onboard deploy."""

    startup_timeout_sec: float = 30.0
    """Maximum time to wait for the first retargeted mocap frame."""

    stale_warn_sec: float = 1.0
    """Warn when no new shared-buffer timestamp is observed for this long."""

    print_every_sec: float = 1.0
    """Progress print interval during the measured run."""


@dataclass
class StreamStats:
    updates: int
    polls: int
    elapsed_sec: float
    producer_dt: list[float]
    arrival_dt: list[float]
    read_ms: list[float]
    lag_ms: list[float]
    stale_events: int


def _percentile_line(name: str, values: list[float], unit: str) -> str:
    if not values:
        return f"  {name:<18}: no samples"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"  {name:<18}: mean={arr.mean():8.3f}{unit}  "
        f"p50={np.percentile(arr, 50):8.3f}{unit}  "
        f"p90={np.percentile(arr, 90):8.3f}{unit}  "
        f"p99={np.percentile(arr, 99):8.3f}{unit}  "
        f"max={arr.max():8.3f}{unit}"
    )


def _stop_retarget_sessions() -> None:
    """Stop retarget subprocesses started by deploy.retarget.start_realtime_retarget."""
    sessions = getattr(retarget_module, "_RETARGET_SESSIONS", [])
    for sess in sessions:
        stop_evt = sess.get("stop_evt")
        if stop_evt is not None:
            stop_evt.set()

    for sess in sessions:
        for key in ("proc", "vis_proc"):
            proc = sess.get(key)
            if proc is None:
                continue
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


def _wait_first_frame(buf, ts, timeout_sec: float) -> float:
    print(f"Waiting for first retargeted Noitom frame (timeout={timeout_sec:.1f}s)...")
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_sec:
        _, mocap_ts = read_mocap_buffer(buf, ts)
        if mocap_ts > 0.0:
            wait_sec = time.perf_counter() - start
            print(f"  First frame received after {wait_sec:.2f}s")
            return mocap_ts
        time.sleep(0.01)
    raise TimeoutError(
        "No retargeted frame arrived before timeout. Check Noitom Axis/PNLink "
        "connection and whether the retarget subprocess printed an error."
    )


def _collect_stream_stats(
    buf,
    ts,
    *,
    duration_sec: float,
    poll_hz: float,
    stale_warn_sec: float,
    print_every_sec: float,
) -> StreamStats:
    poll_dt = 1.0 / poll_hz if poll_hz > 0 else 0.0
    producer_dt: list[float] = []
    arrival_dt: list[float] = []
    read_ms: list[float] = []
    lag_ms: list[float] = []

    polls = 0
    updates = 0
    stale_events = 0
    last_ts = 0.0
    last_arrival_wall = 0.0
    last_progress_t = time.perf_counter()
    last_progress_updates = 0
    last_stale_report_t = 0.0
    start_perf = time.perf_counter()
    next_poll_t = start_perf

    while True:
        now_perf = time.perf_counter()
        elapsed = now_perf - start_perf
        if elapsed >= duration_sec:
            break

        t0 = time.perf_counter()
        _, mocap_ts = read_mocap_buffer(buf, ts)
        read_ms.append((time.perf_counter() - t0) * 1e3)
        polls += 1

        now_wall = time.time()
        if mocap_ts > 0.0 and mocap_ts != last_ts:
            if last_ts > 0.0:
                producer_dt.append(mocap_ts - last_ts)
            if last_arrival_wall > 0.0:
                arrival_dt.append(now_wall - last_arrival_wall)
            lag_ms.append(max(0.0, now_wall - mocap_ts) * 1e3)
            updates += 1
            last_ts = mocap_ts
            last_arrival_wall = now_wall

        stale_for = now_wall - last_arrival_wall
        if (
            last_arrival_wall > 0.0
            and stale_for > stale_warn_sec
            and now_wall - last_stale_report_t > stale_warn_sec
        ):
            stale_events += 1
            last_stale_report_t = now_wall
            print(f"  WARN: no new mocap timestamp for {stale_for:.2f}s")

        if now_perf - last_progress_t >= print_every_sec:
            win_elapsed = now_perf - last_progress_t
            win_updates = updates - last_progress_updates
            print(
                f"  t={elapsed:6.1f}s  update={win_updates / win_elapsed:6.1f}Hz  "
                f"poll={polls / elapsed:6.1f}Hz  total_updates={updates}"
            )
            last_progress_t = now_perf
            last_progress_updates = updates

        if poll_dt > 0.0:
            next_poll_t += poll_dt
            sleep_for = next_poll_t - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            else:
                next_poll_t = time.perf_counter()

    return StreamStats(
        updates=updates,
        polls=polls,
        elapsed_sec=time.perf_counter() - start_perf,
        producer_dt=producer_dt,
        arrival_dt=arrival_dt,
        read_ms=read_ms,
        lag_ms=lag_ms,
        stale_events=stale_events,
    )


def _print_summary(args: BenchNoitomArgs, stats: StreamStats) -> None:
    update_hz = stats.updates / stats.elapsed_sec if stats.elapsed_sec > 0 else 0.0
    poll_hz = stats.polls / stats.elapsed_sec if stats.elapsed_sec > 0 else 0.0
    duplicate_reads = max(0, stats.polls - stats.updates)
    duplicate_pct = 100.0 * duplicate_reads / max(stats.polls, 1)

    print(f"\n{'=' * 88}")
    print("  Noitom online stream benchmark")
    print(f"{'=' * 88}")
    print(f"  duration       : {stats.elapsed_sec:.2f}s")
    print(f"  buffer_ms      : {args.buffer_ms:.1f}")
    print(f"  gmr_rt_pin     : {args.gmr_rt_pin}")
    print(f"  updates        : {stats.updates} ({update_hz:.2f} Hz)")
    print(f"  polls          : {stats.polls} ({poll_hz:.2f} Hz)")
    print(f"  duplicate reads: {duplicate_reads} ({duplicate_pct:.1f}%)")
    print(f"  stale warnings : {stats.stale_events}")

    if stats.producer_dt:
        producer_hz = 1.0 / np.mean(stats.producer_dt)
        print(f"  producer hz    : {producer_hz:.2f} Hz from shared-buffer timestamps")
    print(_percentile_line("producer dt", [v * 1e3 for v in stats.producer_dt], "ms"))
    print(_percentile_line("arrival dt", [v * 1e3 for v in stats.arrival_dt], "ms"))
    print(_percentile_line("read latency", stats.read_ms, "ms"))
    print(_percentile_line("timestamp lag", stats.lag_ms, "ms"))
    print(f"{'=' * 88}")

    if args.buffer_ms > 0:
        print(
            "Note: buffer_ms > 0 measures the jitter-buffer output frequency seen "
            "by play_track_onboard, not the raw Noitom packet arrival frequency."
        )
    else:
        print(
            "Note: buffer_ms = 0 measures direct Noitom -> GMR -> shared-buffer "
            "updates without the jitter-buffer output pacer."
        )


def main(args: BenchNoitomArgs) -> None:
    print_environment_info(extra_packages=["noitom", "gmr"])
    print("\nStarting Noitom PNLink retarget subprocess...")
    print(
        f"  human_height={args.human_height:.2f}  "
        f"buffer_ms={args.buffer_ms:.1f}  gmr_rt_pin={args.gmr_rt_pin}"
    )

    buf_mocap, ts_mocap, _ = start_realtime_retarget(
        robot="unitree_g1",
        dof_full=7 + 29,
        actual_human_height=args.human_height,
        visualize_retarget=False,
        mocap_type="pnlink",
        buffer_ms=args.buffer_ms,
        rt_pin=args.gmr_rt_pin,
    )

    try:
        _wait_first_frame(buf_mocap, ts_mocap, args.startup_timeout_sec)

        if args.warmup_sec > 0.0:
            print(f"Warmup for {args.warmup_sec:.1f}s...")
            _collect_stream_stats(
                buf_mocap,
                ts_mocap,
                duration_sec=args.warmup_sec,
                poll_hz=args.poll_hz,
                stale_warn_sec=args.stale_warn_sec,
                print_every_sec=max(args.warmup_sec + 1.0, 1.0),
            )

        print(f"\nBenchmarking for {args.duration_sec:.1f}s...")
        stats = _collect_stream_stats(
            buf_mocap,
            ts_mocap,
            duration_sec=args.duration_sec,
            poll_hz=args.poll_hz,
            stale_warn_sec=args.stale_warn_sec,
            print_every_sec=args.print_every_sec,
        )
        _print_summary(args, stats)
    finally:
        _stop_retarget_sessions()


if __name__ == "__main__":
    main(tyro.cli(BenchNoitomArgs))
