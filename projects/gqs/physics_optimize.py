"""
Physics-based Filtering via PPO Training + Inference Pipeline.

A completely different approach from physics_filter.py:
1. Train a single PPO policy on the dataset (privileged, no noise, no DR)
2. Use the trained policy to track each motion in simulation
3. Record the tracker's executed motion (physically validated)
4. Filter based on success (full completion) and MPJPE threshold
5. Save the tracker's motion — physically valid and more robust than the original

The key insight: if a trained PPO tracker can successfully reproduce a motion
in simulation with low MPJPE, the motion is physically plausible. The recorded
tracker motion is a "physics-cleaned" version that is guaranteed executable.
"""

import os
import sys
import tyro
import json
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from absl import logging
from jax import tree_util as jtu
from dataclasses import dataclass
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


from utils.logger import LOGGER  # noqa: F401
from tracking.constants import KPT_NAMES
from tracking.convert_qpos2kpt import extract_kpt
from tracking.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config
from tracking.metrics import (
    calculate_kpt_mae_error,
    calculate_trajectory_length,
)

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"


@dataclass
class Args:
    mocap_dir: str = "storage/mocap/amass_train_convert"
    mocap_cache: str = "device"
    output_dir: str = "storage/mocap/amass_train_optimized"
    log_path: str = "storage/log/amass_train_optimized"
    onnx_path: str = "storage/ckpt/amass_train_optimized.onnx"
    result_json: str = "storage/gqs_score/amass_train_optimized.json"
    # Training parameters
    num_timesteps: int = 5_000_000_000
    num_layers: int = 3
    hidden_size: int = 2048
    # Resume training from a pretrained ONNX policy. If None, train from scratch.
    resume_onnx: str | None = None
    # Inference & filtering
    mpjpe_threshold: float = 0.025
    workers: int = 32
    freq: int = 50
    # Pipeline control
    skip_train: bool = False


class _OnnxPolicyArgs:
    """Minimal args object for get_policy_onnx (MLP type)."""

    def __init__(self, onnx_track: str):
        self.onnx_track = onnx_track
        self.policy_type = "mlp"


# ============================================================
# Phase 1: Training
# ============================================================


def run_training(args: Args):
    """Train PPO with privileged info, zero noise, zero DR."""
    logging.info("=" * 60)
    logging.info("Phase 1: Training PPO Policy")
    logging.info("  privileged=True, noise_level=0, dr_level=0")
    logging.info("=" * 60)

    if args.skip_train and Path(args.onnx_path).exists():
        logging.info(f"ONNX exists at {args.onnx_path}, skipping training.")
        return

    if args.resume_onnx is not None and not Path(args.resume_onnx).exists():
        raise FileNotFoundError(f"resume_onnx not found: {args.resume_onnx}")

    env = os.environ.copy()
    env["JAX_PMAP_SHMAP_MERGE"] = "False"

    cmd = [
        sys.executable, "-m", "tracking.train",
        "--mocap_dir", args.mocap_dir,
        "--mocap_cache", args.mocap_cache,
        "--log_path", args.log_path,
        "--onnx_path", args.onnx_path,
        "--num_timesteps", str(args.num_timesteps),
        "--num_layers", str(args.num_layers),
        "--hidden_size", str(args.hidden_size),
        "--noise_level", "0",
        "--dr_level", "0",
        "--privileged",
    ]
    if args.resume_onnx is not None:
        cmd += ["--resume_onnx", args.resume_onnx]
        logging.info(f"Resuming training from ONNX: {args.resume_onnx}")

    logging.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, env=env)
        logging.info(f"Training completed. ONNX saved: {args.onnx_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed (exit code {e.returncode})")
        raise


# ============================================================
# Phase 2: Inference & Filtering
# (follows scripts/eval_parallel.py pattern exactly)
# ============================================================


def _load_npz_with_qpos(file_path: Path) -> Dict:
    data = dict(np.load(file_path, allow_pickle=True))
    if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
        data["qpos"] = np.concatenate(
            [data["root_pos"], data["root_rot"], data["dof_pos"]],
            axis=1,
        )
    if "qpos" not in data:
        raise ValueError(f"{file_path} missing qpos (or root_pos/root_rot/dof_pos) field, cannot convert.")
    return data


def _track_single_traj(
    traj_id: int,
    ref_traj: Dict,
    file_name: str,
    args: Args,
    env_cfg,
    policy=None,
) -> Dict:
    """
    Track a single reference motion using the trained policy.
    Records the tracker's executed motion; if success and MPJPE < threshold,
    converts and saves the tracked motion to output_dir.
    """
    from tracking.policy import get_policy_onnx

    local_policy = policy or get_policy_onnx(_OnnxPolicyArgs(args.onnx_path))

    _init_qpos = ref_traj["qpos"][0].copy()
    _init_qpos[:2] = 0.0
    mj_sim = G1TrackMjSim(init_qpos=_init_qpos, headless=True, ctrl_dt=env_cfg.ctrl_dt)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, local_policy, privileged=True)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    nq, nv = mj_sim.mj_model.nq, mj_sim.mj_model.nv
    traj_len = len(ref_traj["qpos"])

    tracked_qpos = np.empty((traj_len, nq), dtype=np.float32)
    tracked_qvel = np.empty((traj_len, nv), dtype=np.float32)
    traj_metrics = {
        "kpt_pos_errors": [],
        "kpt_rot_errors": [],
        "joint_pos_errors": [],
        "state_history": [],
    }

    for track_step in range(traj_len):
        ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
        track_step_next = np.clip(track_step + 1, 0, traj_len - 1)
        ref_next = jtu.tree_map(lambda x: x[track_step_next][None], ref_traj)

        action = infer_fn.infer_onnx(
            state, {"ref_curr": ref_curr, "ref_next": ref_next}
        )
        state = mj_sim.step(state, action)

        tracked_qpos[track_step] = state.mj_data.qpos
        tracked_qvel[track_step] = state.mj_data.qvel

        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(
            state, ref_curr, ref_next, mj_sim.mj_model
        )
        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
        traj_metrics["state_history"].append({
            "qpos": state.mj_data.qpos.copy(),
            "qvel": state.mj_data.qvel.copy(),
            "xpos": state.mj_data.xpos.copy(),
            "xmat": state.mj_data.xmat.copy(),
        })

    traj_length_ratio, termination_step = calculate_trajectory_length(
        traj_metrics["state_history"], ref_traj, mj_sim.mj_model,
    )
    is_success = traj_length_ratio >= 1.0
    avg_mpjpe = float(np.mean(traj_metrics["kpt_pos_errors"]))
    passed = is_success and avg_mpjpe < args.mpjpe_threshold

    logging.info(
        f"  Traj {traj_id} ({file_name}): "
        f"completion={traj_length_ratio:.4f} ({termination_step}/{traj_len}), "
        f"mpjpe={avg_mpjpe:.6f} m, "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if passed:
        kpt_data = extract_kpt(
            mj_sim.mj_model,
            qpos_src=tracked_qpos,
            qvel_src=tracked_qvel,
            key_body_names=KPT_NAMES,
            fps=args.freq,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / file_name
        np.savez(str(output_path).replace(".npz", ""), **kpt_data)

    return {
        "traj_id": traj_id,
        "file_name": file_name,
        "is_success": is_success,
        "length_ratio": float(traj_length_ratio),
        "termination_step": termination_step,
        "traj_length": traj_len,
        "avg_mpjpe": avg_mpjpe,
        "passed": passed,
    }


def _parallel_worker(args_tuple: Tuple[int, Dict, str, Args]) -> Dict:
    """
    Multi-process worker function that runs completely independently.
    All dependencies must be rebuilt within the process, as processes do not
    share memory. (Same pattern as scripts/eval_parallel.py)
    """
    traj_id, ref_traj, file_name, args = args_tuple
    env_cfg = g1_infer_env_config(ctrl_dt = 1 / args.freq)
    result = _track_single_traj(
        traj_id=traj_id,
        ref_traj=ref_traj,
        file_name=file_name,
        args=args,
        env_cfg=env_cfg,
        policy=None,
    )
    return result


def _save_results(json_path: str, results: List[Dict], args: Args):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "mpjpe_threshold": args.mpjpe_threshold,
            "onnx_path": args.onnx_path,
            "mocap_dir": args.mocap_dir,
            "mocap_cache": args.mocap_cache,
        },
        "results": results,
    }
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.rename(tmp, json_path)


def run_inference_and_filter(args: Args):
    """Infer on every motion in the dataset, filter and save tracker motions."""
    logging.info("=" * 60)
    logging.info("Phase 2: Inference & Filtering")
    logging.info("=" * 60)

    if not Path(args.onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    mocap_dir = Path(args.mocap_dir)
    all_files = sorted(list(mocap_dir.rglob("*.npz")))

    if not all_files:
        raise ValueError(f"No .npz trajectories found under {mocap_dir}.")

    existing_output = {f.name for f in Path(args.output_dir).glob("*.npz")}
    logging.info(f"Total files: {len(all_files)}, Already in output: {len(existing_output)}")

    traj_data: List[Dict] = []
    traj_names: List[str] = []
    for file in tqdm(all_files, desc="Loading trajectories"):
        traj_data.append(_load_npz_with_qpos(file))
        traj_names.append(file.name)

    tasks = [
        (traj_id, traj_data[traj_id], traj_names[traj_id], args)
        for traj_id in range(len(traj_data))
    ]

    all_results: List[Dict] = []
    max_workers = max(1, int(args.workers))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parallel_worker, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference & Filter"):
            all_results.append(fut.result())

    all_results.sort(key=lambda x: x["traj_id"])
    _save_results(args.result_json, all_results, args)

    total = len(all_results)
    success = sum(1 for r in all_results if r.get("is_success", False))
    passed = sum(1 for r in all_results if r.get("passed", False))
    mpjpe_vals = [r["avg_mpjpe"] for r in all_results if r.get("passed")]
    avg_mpjpe_passed = float(np.mean(mpjpe_vals)) if mpjpe_vals else 0.0

    logging.info(f"\n=== Filtering Summary ({total} trajectories) ===")
    logging.info(f"  Success Rate:       {success}/{total} ({100 * success / max(total, 1):.1f}%)")
    logging.info(f"  Passed (saved):     {passed}/{total} ({100 * passed / max(total, 1):.1f}%)")
    logging.info(f"  MPJPE threshold:    {args.mpjpe_threshold:.4f} m")
    logging.info(f"  Avg MPJPE (passed): {avg_mpjpe_passed:.6f} m")
    logging.info(f"  Output:             {args.output_dir}")

    return all_results


# ============================================================
# Main
# ============================================================


def main(args: Args):
    logging.info("=" * 60)
    logging.info("Physics Filter (Training + Tracking Pipeline)")
    logging.info("=" * 60)
    logging.info(f"  Input:           {args.mocap_dir}")
    logging.info(f"  Mocap cache:     {args.mocap_cache}")
    logging.info(f"  Output:          {args.output_dir}")
    logging.info(f"  ONNX:            {args.onnx_path}")
    logging.info(f"  Resume ONNX:     {args.resume_onnx}")
    logging.info(f"  MPJPE threshold: {args.mpjpe_threshold} m")
    logging.info("=" * 60)

    run_training(args)
    run_inference_and_filter(args)

    logging.info("\nPipeline Complete!")


if __name__ == "__main__":
    main(tyro.cli(Args))
