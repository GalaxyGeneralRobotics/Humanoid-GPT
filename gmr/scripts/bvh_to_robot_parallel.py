"""Batch BVH -> robot qpos retargeting for Humanoid-GPT training data.

Ported from GMR-galbot/scripts/bvh_to_robot_batch_rec.py to use the integrated
`gmr` package.  BVH retargeting keeps the original GMR IK weights and the default
Savitzky-Golay smoothing; the IK model's joint ranges are narrowed to a strict
subset of the tracking model's limits, and the smoothed trajectory is clipped back
into them, so the export stays reachable for the trained policy.  Output .npz
carries `qpos` (root7 + 29 dof) and `frequency`; downstream
`tracking.convert_qpos2kpt` recomputes keypoints/qvel on the deploy model.

Input format (Noitom PNS vs OptiTrack/LAFAN) is auto-detected per file, along
with each file's capture fps (from its `Frame Time`); use --assume-format to
force one format if detection ever misfires.

Quality gate (to maximize training-data quality, non-standard captures are
skipped, not retargeted):
  * fps must be within [FPS_MIN, FPS_MAX] (default 90-120);
  * skeleton joint count must be a known-good layout (51=OptiTrack/LAFAN,
    59=Noitom PNS). Other counts (58/60/61/63) are meters-unit export dialects
    (e.g. obstacle_avoidance `objects*` batches) that silently collapse when
    parsed as cm and make GMR hallucinate plausible-but-wrong poses;
  * a runtime safety net validates the pose-invariant canonical leg length from
    BVH OFFSET values, rejecting skeletons incompatible with centimeter units.

Run from the repo root:
    python -m gmr.scripts.bvh_to_robot_parallel \
        --bvh-dir <dir> --out-dir <dir> [--num_proc 8]
"""
import argparse
import os
import sys
import pathlib
import traceback
import numpy as np
import mujoco as mj
from tqdm import tqdm
from scipy import signal
from rich import print
from multiprocessing import Pool

# Make `import gmr` work regardless of how this file is launched.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from gmr import GeneralMotionRetargeting as GMR, ROBOT_XML_DICT
from gmr.sources import SkeletonScaleError, load_lafan1_file

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_SKIP = "skip"

# 29-dof -> 23-dof reduction: dof indices removed (wrist pitch/yaw + ankle roll)
JOINTS_TO_REMOVE_23DOF = [13, 14, 20, 21, 27, 28]

# ---- Quality gate ----
FPS_MIN, FPS_MAX = 90, 120          # keep only captures in this fps range
VALID_NJOINTS = (51, 59)            # 51=OptiTrack/LAFAN, 59=Noitom PNS; other
                                    # counts are meters-unit export dialects


def read_bvh_header(bvh_file, fallback_fps=90):
    """One-pass BVH header scan (no full parse): returns (fps, njoints).

    `njoints` counts ROOT+JOINT declarations (all appear before MOTION), and
    `fps` is derived from `Frame Time`.
    """
    njoints = 0
    fps = fallback_fps
    with open(bvh_file, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("ROOT ") or s.startswith("JOINT "):
                njoints += 1
            elif s.startswith("Frame Time:"):
                ft = float(s.split(":", 1)[1].strip())
                if ft > 0:
                    fps = round(1.0 / ft)
                break
    return fps, njoints


def load_bvh_auto(bvh_file, assume="auto", drop_tpose=3):
    """Load BVH as GMR frames, auto-detecting Noitom PNS vs OptiTrack/LAFAN.

    PNS BVH is space-delimited; OptiTrack/LAFAN uses wider separators, so parsing
    it as PNS raises ValueError -> we fall back. Only PNS carries leading T-pose
    frames, which are dropped. Returns (frames, human_height, pns).
    """
    if assume == "pns":
        order = [True]
    elif assume == "optitrack":
        order = [False]
    else:
        order = [True, False]

    last_err = None
    for pns in order:
        try:
            frames, height = load_lafan1_file(bvh_file, pns=pns)
            if pns and drop_tpose > 0:
                frames = frames[drop_tpose:]
            return frames, height, pns
        except SkeletonScaleError:
            raise
        except ValueError as e:
            last_err = e
    raise ValueError(f"Could not parse BVH as PNS or OptiTrack: {bvh_file} ({last_err})")


def smooth_motion_data(qpos_list, window_length=5, polyorder=2):
    """Smooth motion using Savitzky-Golay filter.

    The root quaternion (columns 3:7) cannot be filtered like the other channels:
    a componentwise low-pass across a sign flip interpolates through the origin,
    and the result is not unit-norm even without one. So make the sign continuous
    first and renormalize after, as `infer_utils.apply_ema_qpos` does.
    """
    if len(qpos_list) < window_length:
        return qpos_list
    if window_length % 2 == 0:
        window_length += 1
    qpos_array = np.array(qpos_list, dtype=np.float64)

    quat = qpos_array[:, 3:7]
    flip = np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], axis=1) < 0, -1.0, 1.0))
    quat[1:] *= flip[:, None]

    smoothed_qpos = np.zeros_like(qpos_array)
    for i in range(qpos_array.shape[1]):
        smoothed_qpos[:, i] = signal.savgol_filter(
            qpos_array[:, i], window_length=window_length, polyorder=polyorder
        )
    norm = np.linalg.norm(smoothed_qpos[:, 3:7], axis=1, keepdims=True)
    smoothed_qpos[:, 3:7] /= np.maximum(norm, 1e-8)
    return smoothed_qpos


def _model_metadata(mj_model):
    """Joint metadata derived directly from the IK model."""
    names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
    return {
        "joint_names": np.array(names, dtype="<U26"),
        "njnt": np.int64(mj_model.njnt),
        "jnt_type": np.array(mj_model.jnt_type, dtype=np.int64),
    }


def process_single_bvh(task):
    full_path, rel_path, args = task
    bvh_file = full_path
    save_root = args["out_dir"]
    robot = args["robot"]
    assume_format = args["assume_format"]
    motion_fps = args["motion_fps"]
    save_23_dof = args["save_23dof"]
    do_smooth = args["smooth"]

    filename = os.path.splitext(os.path.basename(bvh_file))[0]
    dof_flag = "23dof" if save_23_dof else "29dof"
    output_dir = os.path.join(save_root, os.path.dirname(rel_path))
    output_path = os.path.join(output_dir, f"{filename}_{motion_fps}Hz_{dof_flag}.npz")

    try:
        lafan1_data_frames, actual_human_height, pns = load_bvh_auto(
            bvh_file, assume=assume_format
        )

        print(f"[yellow]Processing BVH:[/yellow] {bvh_file}, "
              f"format: {'pns' if pns else 'optitrack'}, fps: {motion_fps}")

        retargeter = GMR(
            src_human="bvh",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
        )

        qpos_list = [retargeter.retarget(frame) for frame in lafan1_data_frames]

        # GMR-galbot export semantics: Savitzky-Golay window 41 (the even input
        # 40 is promoted by smooth_motion_data), polyorder 3.
        if do_smooth:
            qpos_list = smooth_motion_data(qpos_list, window_length=40, polyorder=3)
        qpos_list = np.array(qpos_list)

        mj_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        mj_data = mj.MjData(mj_model)

        # The IK solve respects the model's joint ranges, but Savitzky-Golay
        # overshoots wherever a joint saturates one, so clip before the height fix
        # to keep the exported reference inside the limits the policy can reach.
        lower, upper = mj_model.jnt_range[1:].T
        np.clip(qpos_list[:, 7:], lower, upper, out=qpos_list[:, 7:])

        # ---- Height fix on the same model the IK solved on ----
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]
        dof_pos = qpos_list[:, 7:]
        lowest_height = float("inf")
        for i in range(len(qpos_list)):
            mj_data.qpos[:3] = root_pos[i]
            mj_data.qpos[3:7] = root_rot[i]
            mj_data.qpos[7:] = dof_pos[i]
            mj.mj_forward(mj_model, mj_data)
            lowest_height = min(lowest_height, mj_data.xpos[1:, 2].min())
        root_pos[:, 2] = root_pos[:, 2] - lowest_height
        qpos_list[:, :3] = root_pos

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        meta = _model_metadata(mj_model)
        if save_23_dof:
            qpos_tgt = np.zeros((len(qpos_list), 30))
            qpos_tgt[:, 0:3] = root_pos
            qpos_tgt[:, 3:7] = root_rot
            qpos_tgt[:, 7:30] = np.delete(dof_pos, JOINTS_TO_REMOVE_23DOF, axis=1)
            ndof = 23
        else:
            qpos_tgt = np.zeros((len(qpos_list), 36))
            qpos_tgt[:, 0:3] = root_pos
            qpos_tgt[:, 3:7] = root_rot
            qpos_tgt[:, 7:36] = dof_pos
            ndof = 29

        retarget_data = {
            "qpos": qpos_tgt,
            "qvel": np.zeros((len(qpos_list), 6 + ndof), dtype=np.float64),
            "frequency": motion_fps,
            "split_points": np.array([0, len(qpos_list)], dtype=np.int64),
            **meta,
        }
        np.savez_compressed(output_path, **retarget_data)
        print(f"[cyan]Saved:[/cyan] {output_path}")
        return {"status": STATUS_SUCCESS, "file_path": bvh_file, "output_path": output_path}

    except SkeletonScaleError as e:
        print(f"[magenta]Skip ({e}):[/magenta] {bvh_file}")
        return {"status": STATUS_SKIP, "file_path": bvh_file, "reason": str(e)}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[red]Error processing BVH file:[/red] {bvh_file}")
        print(f"[red]Error:[/red] {error_msg}")
        return {
            "status": STATUS_ERROR,
            "file_path": bvh_file,
            "error_msg": error_msg,
            "traceback": traceback.format_exc(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh-dir", required=True, type=str)
    parser.add_argument("--robot", choices=list(ROBOT_XML_DICT.keys()), default="unitree_g1")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--assume-format", choices=["auto", "pns", "optitrack"],
                        default="auto",
                        help="input BVH format; 'auto' detects Noitom PNS vs "
                             "OptiTrack/LAFAN per file (default)")
    parser.add_argument("--save_23dof", action="store_true", help="save 23 DoF (default 29)")
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply Savitzky-Golay smoothing (default: enabled)",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dof_flag = "23dof" if args.save_23dof else "29dof"

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[blue]GMR model:[/blue] {ROBOT_XML_DICT[args.robot]}")

    total_bvh, skipped, tasks = 0, 0, []
    dropped_fps, dropped_format = 0, 0
    for root, _dirs, files in os.walk(args.bvh_dir):
        for f in files:
            if not f.endswith(".bvh"):
                continue
            total_bvh += 1
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, args.bvh_dir)
            filename = os.path.splitext(f)[0]
            motion_fps, njoints = read_bvh_header(full_path)

            # ---- Quality gate (skip non-standard captures entirely) ----
            if not (FPS_MIN <= motion_fps <= FPS_MAX):
                dropped_fps += 1
                continue
            if njoints not in VALID_NJOINTS:
                dropped_format += 1
                continue

            output_dir = os.path.join(args.out_dir, os.path.dirname(rel_path))
            output_path = os.path.join(output_dir, f"{filename}_{motion_fps}Hz_{dof_flag}.npz")
            if os.path.exists(output_path) and not args.overwrite:
                skipped += 1
                continue
            tasks.append((full_path, rel_path, {
                "out_dir": args.out_dir,
                "robot": args.robot,
                "assume_format": args.assume_format,
                "motion_fps": motion_fps,
                "save_23dof": args.save_23dof,
                "smooth": args.smooth,
            }))

    print(f"[green]Found {total_bvh} BVH files[/green]; to process: {len(tasks)}, "
          f"already-done: {skipped}, "
          f"[yellow]dropped fps∉[{FPS_MIN},{FPS_MAX}]: {dropped_fps}[/yellow], "
          f"[yellow]dropped format(joints∉{VALID_NJOINTS}): {dropped_format}[/yellow]")
    if not tasks:
        print("[green]Nothing to do.[/green]")
        sys.exit(0)

    with Pool(processes=args.num_proc) as pool:
        results = list(tqdm(pool.imap(process_single_bvh, tasks), total=len(tasks)))

    success = sum(r["status"] == STATUS_SUCCESS for r in results)
    failed = [r for r in results if r["status"] == STATUS_ERROR]
    runtime_skipped = [r for r in results if r["status"] == STATUS_SKIP]
    print("\n" + "=" * 60)
    print(f"[green]Success:[/green] {success}  [red]Failed:[/red] {len(failed)}  "
          f"[magenta]Skipped(degenerate):[/magenta] {len(runtime_skipped)}  "
          f"[magenta]Dropped(fps):[/magenta] {dropped_fps}  "
          f"[magenta]Dropped(format):[/magenta] {dropped_format}  "
          f"[magenta]Already-done:[/magenta] {skipped}  Total: {total_bvh}")
    if failed:
        log_path = os.path.join(args.out_dir, "failed_files.log")
        with open(log_path, "w") as fh:
            for r in failed:
                print(f"[red]FAIL[/red] {r['file_path']}: {r['error_msg']}")
                fh.write(f"File: {r['file_path']}\nError: {r['error_msg']}\n{'-'*40}\n")
        print(f"[yellow]Failed list -> {log_path}[/yellow]")
