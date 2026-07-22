import os
import sys
import tyro
import numpy as np
from tqdm import tqdm
from absl import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mujoco
import mujoco.viewer  # noqa: F401
from jax import tree_util as jtu
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.logger import LOGGER  # noqa: F401
from tracking import constants as consts
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.policy import Args, get_policy_onnx
from tracking.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config
from tracking.metrics import (
    calculate_joint_tracking_error,
    calculate_kpt_mae_error,
    calculate_root_tracking_error,
    calculate_trajectory_length,
    calculate_joint_jerk,
)


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"


@dataclass
class ParallelEvalArgs(Args):
    onnx_track: str = "storage/ckpt/amass.onnx"
    track_dir: str = "storage/test/1"
    privileged: bool = False
    num_envs: int = 1
    device: str = "cpu"
    convert: bool = False
    freq: int = 50
    convert_xml_path: str = str(consts.TRACK_XML)
    workers: int = 1


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


def _convert_traj_to_kpt(data: Dict, mj_model: mujoco.MjModel, freq_tgt: int) -> Dict:
    freq_src = float(data["frequency"]) if "frequency" in data else 50
    qpos_src = np.asarray(data["qpos"], dtype=np.float32)
    return qpos2kpt(
        mj_model,
        qpos_src=qpos_src,
        freq_src=freq_src,
        freq_tgt=freq_tgt,
        interp_sec=0.0,
        end_default_sec=0.0,
        debug=False,
        foot_contact_est=False,
        height_clip_mode=None,
        video_path=None,
    )


def _build_policy(args: ParallelEvalArgs):
    if not args.onnx_track.endswith(".onnx"):
        raise ValueError(f"Unsupported onnx_track format: {args.onnx_track} (expected .onnx)")
    return get_policy_onnx(args)


def _evaluate_single_traj(
        traj_id: int,
        ref_traj: Dict,
        file_name: str,
        args: ParallelEvalArgs,
        env_cfg,
        policy=None,
):
    local_policy = policy or _build_policy(args)
    _init_qpos = ref_traj["qpos"][0].copy()
    _init_qpos[:2] = 0.0
    mj_sim = G1TrackMjSim(init_qpos=_init_qpos, headless=True, ctrl_dt=env_cfg.ctrl_dt)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, local_policy, privileged=args.privileged)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    traj_metrics = {
        "kpt_pos_errors": [],
        "kpt_rot_errors": [],
        "joint_pos_errors": [],
        "joint_vel_errors": [],
        "root_pos_errors": [],
        "root_vel_errors": [],
        "root_yaw_errors": [],
        "state_history": [],
    }

    traj_len = len(ref_traj["qpos"])
    for track_step in range(traj_len):
        ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
        track_step_next = np.clip(track_step + 1, 0, traj_len - 1)
        ref_next = jtu.tree_map(lambda x: x[track_step_next][None], ref_traj)

        action = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = mj_sim.step(state, action)

        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(state, ref_curr, ref_next, mj_sim.mj_model)
        joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(state, ref_curr)
        root_pos_err_mm, root_vel_err_mms, root_yaw_err = calculate_root_tracking_error(state, ref_curr)

        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
        traj_metrics["joint_pos_errors"].append(joint_pos_mae)
        traj_metrics["joint_vel_errors"].append(joint_vel_mae)
        traj_metrics["root_pos_errors"].append(root_pos_err_mm)
        traj_metrics["root_vel_errors"].append(root_vel_err_mms)
        traj_metrics["root_yaw_errors"].append(root_yaw_err)
        traj_metrics["state_history"].append(
            {
                "qpos": state.mj_data.qpos.copy(),
                "qvel": state.mj_data.qvel.copy(),
                "xpos": state.mj_data.xpos.copy(),
                "xmat": state.mj_data.xmat.copy(),
            }
        )

    actual_trajectory_length = len(ref_traj["qpos"])
    traj_length_ratio, termination_step = calculate_trajectory_length(
        traj_metrics["state_history"],
        ref_traj,
        mj_sim.mj_model,
    )
    avg_kpt_pos_error = np.mean(traj_metrics["kpt_pos_errors"])
    avg_kpt_rot_error = np.mean(traj_metrics["kpt_rot_errors"])
    avg_joint_pos_error = np.mean(traj_metrics["joint_pos_errors"])
    avg_joint_vel_error = np.mean(traj_metrics["joint_vel_errors"])
    avg_root_pos_error = np.mean(traj_metrics["root_pos_errors"])
    avg_root_vel_error = np.mean(traj_metrics["root_vel_errors"])
    avg_root_yaw_error = np.mean(traj_metrics["root_yaw_errors"])
    avg_joint_jerk = calculate_joint_jerk(traj_metrics["state_history"], env_cfg.ctrl_dt)

    logging.info(f"  Trajectory {traj_id} completed:")
    logging.info(f"    Completion: {traj_length_ratio:.4f} ({termination_step}/{actual_trajectory_length} steps)")
    logging.info(f"    KPT Position MAE: {avg_kpt_pos_error:.6f} m")
    logging.info(f"    KPT Rotation MAE: {avg_kpt_rot_error:.6f} rad")
    logging.info(f"    Joint Position MAE: {avg_joint_pos_error:.6f} rad")
    logging.info(f"    Joint Velocity MAE: {avg_joint_vel_error:.6f} rad/s")
    logging.info(f"    Root Pos Error: {avg_root_pos_error:.3f} mm")
    logging.info(f"    Root Vel Error: {avg_root_vel_error:.3f} mm/s")
    logging.info(f"    Root Yaw Error: {avg_root_yaw_error:.6f} rad")
    logging.info(f"    Joint Jerk: {avg_joint_jerk:.3f} rad/s^3")

    return {
            "traj_id": traj_id,
            "file_name": file_name,
            "length_ratio": traj_length_ratio,
            "kpt_pos_mae": avg_kpt_pos_error,
            "kpt_rot_mae": avg_kpt_rot_error,
            "joint_pos_mae": avg_joint_pos_error,
            "joint_vel_mae": avg_joint_vel_error,
            "root_pos_err_mm": avg_root_pos_error,
            "root_vel_err_mms": avg_root_vel_error,
            "root_yaw_err": avg_root_yaw_error,
            "joint_jerk": avg_joint_jerk,
        }


def _parallel_worker(args_tuple: Tuple[int, Dict, str, ParallelEvalArgs]) -> Dict:
    """
    Multi-process worker function that runs completely independently.
    All dependencies must be rebuilt within the process, as processes do not share memory.
    """
    traj_id, ref_traj, file_name, args = args_tuple

    # Recreate env_cfg within the process as it may not be serializable
    env_cfg = g1_infer_env_config(ctrl_dt = 1 / args.freq)

    metric = _evaluate_single_traj(
        traj_id=traj_id,
        ref_traj=ref_traj,
        file_name=file_name,
        args=args,
        env_cfg=env_cfg,
        policy=None
    )
    return metric


def main(args: ParallelEvalArgs):

    data_path = Path(args.track_dir)
    if data_path.is_file():
        traj_files = [data_path]
    elif data_path.is_dir():
        traj_files = sorted(list(data_path.rglob("*.npz")))
    else:
        raise ValueError(f"{data_path} not exist.")

    if not traj_files:
        raise ValueError(f"No .npz reference trajectories found under {data_path}.")

    convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    traj_data: List[Dict] = []
    traj_names: List[str] = []
    for file in traj_files:
        raw_data = _load_npz_with_qpos(file)
        if args.convert:
            traj_data.append(_convert_traj_to_kpt(raw_data, convert_mj_model, args.freq))
        else:
            traj_data.append(raw_data)
        traj_names.append(file.name)

    all_metrics: List[Dict] = []
    tasks = [
        (traj_id, traj_data[traj_id], traj_names[traj_id], args)
        for traj_id in range(len(traj_data))
    ]

    max_workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parallel_worker, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel Eval"):
            all_metrics.append(fut.result())

    # Sort by traj_id to ensure consistent result order
    all_metrics.sort(key=lambda x: x["traj_id"])

    if all_metrics:
        logging.info(f"\n=== Overall Summary ({len(all_metrics)} trajectories) ===")
        avg_length_ratio = np.mean([m["length_ratio"] for m in all_metrics])
        avg_kpt_pos = np.mean([m["kpt_pos_mae"] for m in all_metrics])
        avg_kpt_rot = np.mean([m["kpt_rot_mae"] for m in all_metrics])
        avg_joint_pos = np.mean([m["joint_pos_mae"] for m in all_metrics])
        avg_joint_vel = np.mean([m["joint_vel_mae"] for m in all_metrics])
        avg_root_pos = np.mean([m["root_pos_err_mm"] for m in all_metrics])
        avg_root_vel = np.mean([m["root_vel_err_mms"] for m in all_metrics])
        avg_root_yaw = np.mean([m["root_yaw_err"] for m in all_metrics])
        avg_joint_jerk = np.mean([m["joint_jerk"] for m in all_metrics])

        # Calculate Success Rate (trajectories with length_ratio >= 1.0 are considered successful)
        successful_trajs = sum(1 for m in all_metrics if m["length_ratio"] >= 1.0)
        success_rate = successful_trajs / len(all_metrics) if all_metrics else 0.0

        logging.info(f"Success Rate: {success_rate:.4f} ({successful_trajs}/{len(all_metrics)} trajectories completed)")
        logging.info(f"Average Trajectory Completion: {avg_length_ratio:.4f} (0-1, 1.0=completed)")
        logging.info(f"Average KPT Position MAE: {avg_kpt_pos:.6f} m")
        logging.info(f"Average KPT Rotation MAE: {avg_kpt_rot:.6f} rad")
        logging.info(f"Average Joint Position MAE: {avg_joint_pos:.6f} rad")
        logging.info(f"Average Joint Velocity MAE: {avg_joint_vel:.6f} rad/s")
        logging.info(f"Average Root Pos Error: {avg_root_pos:.3f} mm")
        logging.info(f"Average Root Vel Error: {avg_root_vel:.3f} mm/s")
        logging.info(f"Average Root Yaw Error: {avg_root_yaw:.6f} rad")
        logging.info(f"Average Joint Jerk: {avg_joint_jerk:.3f} rad/s^3")

    return all_metrics


if __name__ == "__main__":
    main(tyro.cli(ParallelEvalArgs))
