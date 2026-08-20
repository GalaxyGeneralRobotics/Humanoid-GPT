"""Benchmark ONLY the ONNX inference call.

Measures `policy.infer(obs)` on dummy observations -- nothing else.
Useful to compare ORT/TRT execution kernel speed across environments.

For full pipeline latency without a robot, see ``bench_offline.py``.
For full pipeline latency on a connected robot, see ``bench_online.py``.

Usage:
    python -m deploy.onboard_deploy.bench_onnx_infer
    python -m deploy.onboard_deploy.bench_onnx_infer --use-trt
    python -m deploy.onboard_deploy.bench_onnx_infer --num-iters 1000
"""

from __future__ import annotations

from deploy.onboard_deploy._bench_utils import aarch64_preload

aarch64_preload()

from dataclasses import dataclass

import numpy as np
import tyro

from deploy.onboard_deploy._bench_utils import (
    BenchProfiler,
    print_environment_info,
)
from tracking.policy import Args as PolicyArgs
from tracking.policy import get_policy_onnx
from tracking.infer_utils import NUM_STATE


@dataclass
class BenchArgs:
    """ONNX-only inference benchmark."""

    onnx_track: str = "storage/ckpts/pns_wo_priv264.onnx"
    policy_type: str = "mlp"
    use_trt: bool = False
    num_warmup: int = 50
    num_iters: int = 500
    device: str = "cuda:0"


def main(args: BenchArgs) -> None:
    print_environment_info()

    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
        policy_type=args.policy_type,
        device=args.device,
    )
    policy = get_policy_onnx(policy_args, use_trt=args.use_trt, strict_trt=False)

    dummy_obs = np.random.randn(1, NUM_STATE).astype(np.float32)

    print(f"\nWarmup ({args.num_warmup} iters)...")
    for _ in range(args.num_warmup):
        policy.infer(dummy_obs)

    prof = BenchProfiler()
    print(f"Benchmarking ({args.num_iters} iters)...")
    for _ in range(args.num_iters):
        with prof.time("onnx_run"):
            policy.infer(dummy_obs)

    header = (
        f"ONNX-only latency  |  model={args.onnx_track}  "
        f"type={args.policy_type}  TRT={args.use_trt}  device={args.device}"
    )
    print(prof.summary(header, iters=args.num_iters))

    arr = np.asarray(prof.timings["onnx_run"])
    print(f"  Throughput (mean):       {1000 / arr.mean():.0f} Hz")
    print(f"  Margin to 50Hz at mean:  {20 - arr.mean():+.2f} ms")
    print(f"  Margin to 50Hz at p99:   {20 - np.percentile(arr, 99):+.2f} ms")
    print()


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
