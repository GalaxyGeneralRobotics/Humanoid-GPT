"""
Physics-based Quality Scoring and Filtering Pipeline (Multi-GPU Version).

Pipeline:
1. Score all motion files using physics-based metrics
2. Save scores to JSON file
3. Filter and copy motions that pass the threshold

Scoring Weights:
- Floating: 200.0 (Top Priority)
- Velocity: 5.0 (High Priority within valid range)
- Foot Sliding: 1.0 (Base Priority)
- Penetration: 10.0 (Reduced)
- Collision: 0.01 (Ignored)
- Jerk: 0.01 (Ignored)
"""

import gc
import os
import sys
import tyro
import time
import json
import math
import shutil
import signal
from pathlib import Path
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from tracking import constants as consts


def count_contacts_exclude(
    data: mjx.Data, geom_id_exclude: int, only_colliding: bool = True
) -> jax.Array:
    """Count contact points whose pair does NOT include the exclude geom."""
    geom_pairs = data.contact.geom  # [N, 2] int
    not_target = (geom_pairs != geom_id_exclude).all(axis=1)
    n = geom_pairs.shape[0]
    active = jnp.arange(n) < data.ncon

    mask = not_target & active
    if only_colliding:
        mask = mask & (data.contact.dist < 0)

    return jnp.sum(mask)


class PhysicsScoringEnv:
    """Minimal MJX model holder for physics-feasibility scoring.

    ``physics_filter`` only needs the MJX model, the per-joint velocity
    limits, and a few body/geom ids to evaluate contacts. We build them
    directly from the G1 ``track`` scene so this stage stays self-contained
    and does not depend on the full JAX training environment.
    """

    def __init__(self, xml_path: str | None = None):
        mj_model = mujoco.MjModel.from_xml_path(
            str(xml_path) if xml_path is not None else str(consts.TRACK_XML)
        )
        self.mjx_model = mjx.put_model(mj_model)
        self.dof_vel_limit = jnp.array(consts.DOF_VEL_LIMITS)
        self.body_id_ankle_l = mj_model.body("left_ankle_roll_link").id
        self.body_id_ankle_r = mj_model.body("right_ankle_roll_link").id
        self.geom_id_floor = mj_model.geom("floor").id


@dataclass
class Args:
    # Input/output paths
    mocap_dir: str = "storage/mocap/amass_train_convert"
    score_json_path: str = "storage/gqs_score/amass_train_filtered.json"
    output_dir: str = "storage/mocap/amass_train_filtered"
    # Scoring parameters
    dt: float = 0.02
    num_gpus: int = -1  # -1 for auto-detection
    min_duration: float = 0.5  # Minimum motion duration threshold (seconds)
    # Filtering parameters
    threshold: float = 90.0


# Global variables
_processes: Optional[List[mp.Process]] = None
_queue: Optional[mp.Queue] = None
_interrupted: bool = False


def signal_handler(signum, frame):
    global _processes, _queue, _interrupted
    if _interrupted:
        sys.exit(1)
    _interrupted = True
    print("\nStopping child processes...")
    if _processes:
        for p in _processes:
            if p.is_alive(): p.terminate()
        for p in _processes:
            p.join(timeout=2)
            if p.is_alive(): p.kill()
    print("Stopped.")
    raise KeyboardInterrupt


def load_motion_file(file_path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        data = dict(np.load(str(file_path), allow_pickle=True))
    except Exception:
        return None
    if "qpos" not in data:
        return None
    if "qvel" not in data:
        qpos = data["qpos"]
        data["qvel"] = np.zeros((len(qpos), qpos.shape[1] - 1))
    return data


# ==========================================
# JIT / VMAP Function Definitions (Contact-based Detection)
# ==========================================

def compute_single_frame(
    sys: mjx.Model,
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    prev_qvel: jnp.ndarray,
    body_id_l: int,
    body_id_r: int,
    geom_id_floor: int,
    dof_vel_limit: jnp.ndarray,
    dt: float
) -> Tuple[float, float, float, float, float, float]:

    d = mjx.make_data(sys)
    d = d.replace(qpos=qpos, qvel=qvel)
    # mjx.forward performs collision detection and fills d.contact
    d = mjx.forward(sys, d)

    # --- 1. Foot Sliding ---
    left_vel = d.cvel[body_id_l][3:5]
    right_vel = d.cvel[body_id_r][3:5]
    l_speed = jnp.linalg.norm(left_vel)
    r_speed = jnp.linalg.norm(right_vel)

    # Simple foot contact detection for sliding calculation
    l_contact = d.xpos[body_id_l][2] < 0.05
    r_contact = d.xpos[body_id_r][2] < 0.05

    p_slide = 0.0
    p_slide += jnp.where(l_contact, jnp.maximum(0.0, l_speed - 0.1), 0.0)
    p_slide += jnp.where(r_contact, jnp.maximum(0.0, r_speed - 0.1), 0.0)

    # --- 2. Velocity Limit ---
    joint_vels = qvel[-len(dof_vel_limit):]
    p_vel = jnp.mean(jnp.maximum(0.0, jnp.abs(joint_vels) - dof_vel_limit))

    # --- 3. Self Collision ---
    n_con = count_contacts_exclude(d, geom_id_floor, only_colliding=True)
    p_col = jnp.clip(n_con, 0.0, 10.0)

    # --- 4. Jerk ---
    accel = (qvel - prev_qvel) / dt
    p_jerk = jnp.linalg.norm(accel)

    # --- 5 & 6. Global Contact Analysis (Any Body Part) ---
    floor_mask = (d.contact.geom1 == geom_id_floor) | (d.contact.geom2 == geom_id_floor)
    dists_to_floor = jnp.where(floor_mask, d.contact.dist, 100.0)
    min_dist = jnp.min(dists_to_floor)

    # [Penetration Detection]
    p_pen = jnp.maximum(0.0, -min_dist - 0.01)

    # [Floating Detection]
    is_air = jnp.where(min_dist > 0.05, 1.0, 0.0)

    return p_slide, p_vel, p_col, p_jerk, p_pen, is_air


@partial(jax.jit, static_argnums=(5, 6, 7, 9))
def compute_clip_metrics_jit(
    sys: mjx.Model,
    qpos_seq: jnp.ndarray,
    qvel_seq: jnp.ndarray,
    prev_qvel_seq: jnp.ndarray,
    mask: jnp.ndarray,
    body_id_l: int,
    body_id_r: int,
    geom_id_floor: int,
    dof_vel_limit: jnp.ndarray,
    dt: float,
):
    vmap_fn = jax.vmap(
        compute_single_frame,
        in_axes=(None, 0, 0, 0, None, None, None, None, None)
    )

    s, v, c, j, p, air = vmap_fn(
        sys, qpos_seq, qvel_seq, prev_qvel_seq,
        body_id_l, body_id_r, geom_id_floor, dof_vel_limit, dt
    )

    mask = mask.astype(s.dtype)

    total_slide = jnp.sum(s * mask)
    total_vel = jnp.sum(v * mask)
    total_col = jnp.sum(c * mask)
    total_jerk = jnp.sum(j * mask)
    total_pen = jnp.sum(p * mask)

    # --- Long-term Floating Detection ---
    valid_air = air * mask
    window_size = int(1.0 / dt)
    kernel = jnp.ones(window_size)
    conv_res = jnp.convolve(valid_air, kernel, mode='same')

    floating_violation_frames = jnp.where(conv_res >= (window_size - 0.1), 1.0, 0.0)
    total_float_frames = jnp.sum(floating_violation_frames * mask)

    return total_slide, total_vel, total_col, total_jerk, total_pen, total_float_frames


def get_padded_batch(qpos, qvel, prev_qvel, chunk_size=512):
    n_frames = qpos.shape[0]
    if n_frames == 0: return None, None, None, None

    target_len = math.ceil(n_frames / chunk_size) * chunk_size
    pad_len = target_len - n_frames

    mask = np.concatenate([np.ones(n_frames), np.zeros(pad_len)])
    qpos_pad = np.pad(qpos, ((0, pad_len), (0, 0)), mode='edge')
    qvel_pad = np.pad(qvel, ((0, pad_len), (0, 0)), mode='constant')
    prev_qvel_pad = np.pad(prev_qvel, ((0, pad_len), (0, 0)), mode='constant')

    return qpos_pad, qvel_pad, prev_qvel_pad, mask


def score_one_file(fpath, env, args):
    mdata = load_motion_file(fpath)
    if mdata is None: return 0.0, {}

    qpos_np = mdata["qpos"]
    qvel_np = mdata["qvel"]
    n_frames = len(qpos_np)

    # === [Hard Filter 1] Duration Check ===
    duration = n_frames * args.dt
    if n_frames < 5 or duration < args.min_duration:
        zero_metrics = {
            "foot_sliding": 100.0, "velocity_violation": 100.0,
            "self_collision": 100.0, "jerk": 100.0,
            "penetration": 100.0, "floating_frames_ratio": 1.0,
            "is_too_short": 1.0
        }
        return 0.0, zero_metrics

    prev_qvel_seq_np = np.concatenate([qvel_np[:1], qvel_np[:-1]])

    qpos_pad, qvel_pad, prev_qvel_pad, mask = get_padded_batch(
        qpos_np, qvel_np, prev_qvel_seq_np, chunk_size=512
    )

    qpos_j = jnp.array(qpos_pad)
    qvel_j = jnp.array(qvel_pad)
    prev_qvel_j = jnp.array(prev_qvel_pad)
    mask_j = jnp.array(mask)
    vlim = env.dof_vel_limit

    t_slide, t_vel, t_col, t_jerk, t_pen, t_float = compute_clip_metrics_jit(
        env.mjx_model,
        qpos_j, qvel_j, prev_qvel_j, mask_j,
        env.body_id_ankle_l, env.body_id_ankle_r, env.geom_id_floor,
        vlim, args.dt
    )

    metrics = {
        "foot_sliding": float(t_slide) / n_frames,
        "velocity_violation": float(t_vel) / n_frames,
        "self_collision": float(t_col) / n_frames,
        "jerk": float(t_jerk) / n_frames,
        "penetration": float(t_pen) / n_frames,
        "floating_frames_ratio": float(t_float) / n_frames
    }

    # === [Soft Scoring] Calibrated Weights (sensitivity analysis) ===
    score = 100.0 - (
        1.70 * metrics["foot_sliding"] +            # Toxic
        44.22 * metrics["velocity_violation"] +     # Friendly
        0.17 * metrics["self_collision"] +          # Friendly
        0.28 * metrics["jerk"] +                    # Neutral
        216.62 * metrics["penetration"] +           # Neutral (safety constraint)
        24.19 * metrics["floating_frames_ratio"]    # Toxic
    )

    return max(0.0, score), metrics


def worker_process(gpu_id: int, file_subset: List[Path], args: Args, return_queue: mp.Queue, launch_delay: float = 0.0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # Per-worker isolated tmpdir, prevents ptxas tempfile race between workers
    # (root cause of "Missing .version directive" ptxas crashes when many JAX
    # processes start simultaneously and share /tmp).
    worker_tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f"physics_filter_gpu{gpu_id}_{os.getpid()}"
    worker_tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(worker_tmp)

    # Stagger startup to reduce contention on ptxas and GPU init.
    if launch_delay > 0:
        time.sleep(launch_delay)

    print(f"[GPU {gpu_id}] Started. Files: {len(file_subset)}  tmpdir={worker_tmp}", flush=True)

    file_names_remaining = [f.name for f in file_subset]

    try:
        env = PhysicsScoringEnv()

        # Warmup
        dummy_q = jnp.zeros((512, env.mjx_model.nq))
        dummy_v = jnp.zeros((512, env.mjx_model.nv))
        dummy_mask = jnp.ones(512)
        _ = compute_clip_metrics_jit(
            env.mjx_model, dummy_q, dummy_v, dummy_v, dummy_mask,
            env.body_id_ankle_l, env.body_id_ankle_r, env.geom_id_floor,
            env.dof_vel_limit, 0.02
        )

        batch_size = 100
        for idx, f in enumerate(tqdm(file_subset, position=gpu_id, desc=f"GPU {gpu_id}")):
            try:
                score, mets = score_one_file(f, env, args)
                return_queue.put(('result', f.name, score, mets))
                file_names_remaining.remove(f.name)
                if (idx + 1) % batch_size == 0:
                    return_queue.put(('batch', gpu_id, idx + 1, None))
                if (idx + 1) % 500 == 0: gc.collect()
            except Exception as e:
                return_queue.put(('error', f.name, None, str(e)))
                file_names_remaining.remove(f.name)

        return_queue.put(('done', gpu_id, [], None))
    except Exception as e:
        err_msg = f"[GPU {gpu_id}] Fatal in worker: {type(e).__name__}: {e}"
        print(err_msg, flush=True)
        # Report remaining files so main process can retry them.
        return_queue.put(('done', gpu_id, file_names_remaining, err_msg))


def load_existing_results(score_json_path: str):
    res, det = {}, {}
    path = Path(score_json_path)
    if path.exists():
        try:
            with open(path, 'r') as f: data = json.load(f)
            if "summary" in data: res = {k: v for k, v in data["summary"]}
            if "details" in data: det = data["details"]
            print(f"Loaded {len(res)} existing results.")
        except: pass
    return res, det


def save_results(path, res, det, new_res, new_det):
    final_res = {**res, **new_res}
    final_det = {**det, **new_det}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "summary": sorted(final_res.items(), key=lambda x: x[1], reverse=True),
        "details": final_det
    }
    tmp = path + ".tmp"
    with open(tmp, 'w') as f: json.dump(out, f, indent=2)
    os.rename(tmp, path)
    return final_res


def run_scoring(args: Args) -> Dict[str, float]:
    """
    Run multi-GPU scoring on all motion files.
    Returns: {filename: score} mapping
    """
    print("=" * 60)
    print("Phase 1: Physics-based Quality Scoring")
    print("=" * 60)

    existing_res, existing_det = load_existing_results(args.score_json_path)
    mocap_dir = Path(args.mocap_dir)
    all_files = sorted(list(mocap_dir.rglob("*.npz")))
    files = [f for f in all_files if f.name not in existing_res]

    print(f"Total: {len(all_files)}, Already scored: {len(existing_res)}, Remaining: {len(files)}")

    if not files:
        print("All files already scored. Skipping scoring phase.")
        return existing_res

    if args.num_gpus == -1:
        try:
            import subprocess
            output = subprocess.check_output("nvidia-smi -L", shell=True).decode().strip()
            avail_gpus = len(output.split("\n"))
        except: avail_gpus = 1
    else: avail_gpus = args.num_gpus

    print(f"Launching on {avail_gpus} GPUs.")
    chunk_size = math.ceil(len(files) / avail_gpus)
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    global _processes, _queue
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    processes = []
    _processes = processes
    _queue = queue

    new_res, new_det = {}, {}
    done_workers = set()
    failed_files: List[str] = []  # filenames whose worker crashed before processing them
    error_files: List[Tuple[str, str]] = []  # individual per-file errors (name, error_str)
    final_scores = existing_res.copy()

    name_to_path = {f.name: f for f in files}

    try:
        # Stagger worker startup by a few seconds each to avoid simultaneous
        # JAX/XLA initialization, which previously triggered ptxas tempfile
        # race conditions ("Missing .version directive at start of file").
        startup_stagger = 3.0
        for i in range(min(avail_gpus, len(chunks))):
            delay = i * startup_stagger
            p = mp.Process(target=worker_process, args=(i, chunks[i], args, queue, delay))
            p.start()
            processes.append(p)

        last_save = time.time()
        while len(done_workers) < len(processes):
            try:
                msg = queue.get(timeout=1.0)
                if msg[0] == 'result':
                    new_res[msg[1]] = msg[2]
                    new_det[msg[1]] = msg[3]
                elif msg[0] == 'batch':
                    if time.time() - last_save > 60:
                        save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
                        last_save = time.time()
                        print(f"Saved. New: {len(new_res)}")
                elif msg[0] == 'error':
                    # Per-file error - just record it
                    error_files.append((msg[1], msg[3] if len(msg) > 3 else ''))
                elif msg[0] == 'done':
                    done_workers.add(msg[1])
                    remaining = msg[2] if len(msg) > 2 and msg[2] else []
                    if remaining:
                        failed_files.extend(remaining)
                        err_str = msg[3] if len(msg) > 3 and msg[3] else 'unknown error'
                        print(f"[GPU {msg[1]}] WORKER FAILED. {len(remaining)} files left unprocessed. ({err_str})")
            except:
                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in done_workers:
                        done_workers.add(i)
                        print(f"[GPU {i}] Process died without sending 'done'. Files in its chunk will be retried.")

        for p in processes: p.join()

        # Detect silently-lost files: anything in the input but neither in
        # new_res nor in failed_files / error_files.
        input_names = set(name_to_path.keys())
        accounted = set(new_res.keys()) | set(failed_files) | set(name for name, _ in error_files)
        silently_lost = input_names - accounted
        if silently_lost:
            print(f"WARNING: {len(silently_lost)} files silently lost (no result, no error). Will retry.")
            failed_files.extend(silently_lost)

        # Persist what we have so far before retrying.
        final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
        print(f"Parallel phase done. Scored so far: {len(final_scores)}, "
              f"failed/lost: {len(failed_files)}, per-file errors: {len(error_files)}")

        # --- Recovery pass: serial retry for files lost due to worker crashes ---
        if failed_files:
            print("\n" + "=" * 60)
            print(f"Recovery pass: serially re-scoring {len(failed_files)} unprocessed files on GPU 0")
            print("=" * 60)
            # Run inside a child process so its CUDA/JAX state is fresh and
            # cannot affect the main process.
            retry_queue = mp.Queue()
            retry_chunk = [name_to_path[n] for n in failed_files if n in name_to_path]
            rp = mp.Process(target=worker_process, args=(0, retry_chunk, args, retry_queue, 0.0))
            rp.start()
            retry_done = False
            while not retry_done:
                try:
                    msg = retry_queue.get(timeout=1.0)
                    if msg[0] == 'result':
                        new_res[msg[1]] = msg[2]
                        new_det[msg[1]] = msg[3]
                    elif msg[0] == 'done':
                        retry_done = True
                        remaining = msg[2] if len(msg) > 2 and msg[2] else []
                        if remaining:
                            print(f"Recovery still missing {len(remaining)} files (worker also crashed).")
                except:
                    if not rp.is_alive():
                        retry_done = True
            rp.join()
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
            print(f"Recovery complete. Total scored: {len(final_scores)}")
        else:
            print(f"Scoring complete. Total scored: {len(final_scores)}")

    except KeyboardInterrupt:
        if new_res:
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
        raise
    except Exception as e:
        print(f"Error: {e}")
        if new_res:
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
    finally:
        _processes = None

    return final_scores


def run_filtering(args: Args, score_map: Dict[str, float]):
    """
    Filter and copy motions that pass the threshold.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Filtering and Copying Passed Motions")
    print("=" * 60)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning source directory: {args.mocap_dir}...")
    src_files = list(Path(args.mocap_dir).rglob("*.npz"))

    print(f"Found {len(src_files)} source files.")
    print(f"Filtering with threshold >= {args.threshold}...")

    passed_count = 0
    already_copied_count = 0
    skipped_count = 0
    missing_score_count = 0

    for src_path in tqdm(src_files, desc="Copying passed motions"):
        file_name = src_path.name
        score = score_map.get(file_name)

        if score is None:
            # If score not found, skip by default
            missing_score_count += 1
            continue

        if score >= args.threshold:
            dst_path = os.path.join(args.output_dir, file_name)
            # Skip copy if destination already exists with matching size (idempotent reruns).
            try:
                if os.path.exists(dst_path) and os.path.getsize(dst_path) == src_path.stat().st_size:
                    already_copied_count += 1
                else:
                    shutil.copy2(src_path, dst_path)
                    passed_count += 1
            except OSError:
                shutil.copy2(src_path, dst_path)
                passed_count += 1
        else:
            skipped_count += 1

    print("-" * 50)
    print(f"Filtering Complete.")
    print(f"  Threshold:           {args.threshold}")
    print(f"  Total Source:        {len(src_files)}")
    print(f"  Newly Copied:        {passed_count}")
    print(f"  Already Copied:      {already_copied_count}")
    print(f"  Total Passed:        {passed_count + already_copied_count}")
    print(f"  Rejected:            {skipped_count}")
    print(f"  Missing Score:       {missing_score_count}")
    print(f"Output Directory:      {args.output_dir}")
    print("-" * 50)


def main(args: Args):
    print("=" * 60)
    print("Physics-based Quality Scoring and Filtering Pipeline")
    print("=" * 60)
    print(f"Input Directory:   {args.mocap_dir}")
    print(f"Score JSON Path:   {args.score_json_path}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Threshold:         {args.threshold}")
    print("=" * 60)

    # Phase 1: Scoring
    score_map = run_scoring(args)

    # Phase 2: Filtering
    run_filtering(args, score_map)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
