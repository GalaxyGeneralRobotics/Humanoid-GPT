"""Batch BVH -> robot qpos retargeting for Humanoid-GPT training data.

Ported from GMR-galbot/scripts/bvh_to_robot_batch_rec.py to use the integrated
`gmr` package, so the produced training data is retargeted onto the *same*
robot model Humanoid-GPT trains/deploys on (storage/assets/unitree_g1_<ver>/
g1_mocap_track.xml). Output .npz carries `qpos` (root7 + 29 dof) and
`frequency`; downstream `tracking.convert_qpos2kpt` recomputes keypoints/qvel
on the deploy model.

Run from the repo root:
    python -m gmr.scripts.bvh_to_robot_batch_rec \
        --bvh_file_folder <dir> --save_path <dir> [--pns] [--num_proc 8]
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
from gmr.sources import load_lafan1_file

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# 29-dof -> 23-dof reduction: dof indices removed (wrist pitch/yaw + ankle roll)
JOINTS_TO_REMOVE_23DOF = [13, 14, 20, 21, 27, 28]


def smooth_motion_data(qpos_list, window_length=5, polyorder=2):
    """Smooth motion using Savitzky-Golay filter."""
    if len(qpos_list) < window_length:
        return qpos_list
    if window_length % 2 == 0:
        window_length += 1
    qpos_array = np.array(qpos_list)
    smoothed_qpos = np.zeros_like(qpos_array)
    for i in range(qpos_array.shape[1]):
        smoothed_qpos[:, i] = signal.savgol_filter(
            qpos_array[:, i], window_length=window_length, polyorder=polyorder
        )
    return smoothed_qpos


def _model_metadata(mj_model):
    """Joint metadata derived straight from the (5010) retarget model so the
    saved npz stays consistent with the deploy kinematics."""
    names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
    return {
        "joint_names": np.array(names, dtype="<U26"),
        "njnt": np.int64(mj_model.njnt),
        "jnt_type": np.array(mj_model.jnt_type, dtype=np.int64),
    }


def process_single_bvh(task):
    full_path, rel_path, args = task
    bvh_file = full_path
    save_root = args["save_path"]
    robot = args["robot"]
    pns = args["pns"]
    motion_fps = args["motion_fps"]
    save_23_dof = args["save_23dof"]
    do_smooth = args["smooth"]

    filename = os.path.splitext(os.path.basename(bvh_file))[0]
    dof_flag = "23dof" if save_23_dof else "29dof"
    output_dir = os.path.join(save_root, os.path.dirname(rel_path))
    output_path = os.path.join(output_dir, f"{filename}_{motion_fps}Hz_{dof_flag}.npz")

    try:
        print(f"[yellow]Processing BVH:[/yellow] {bvh_file}, pns: {pns}")
        lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file, pns=pns)
        if pns:
            # First few frames are the default T-pose; drop them.
            lafan1_data_frames = lafan1_data_frames[3:]

        retargeter = GMR(
            src_human="bvh",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
        )

        qpos_list = [retargeter.retarget(frame) for frame in lafan1_data_frames]

        # No smoothing by default: a savgol low-pass blunts fast peaks and
        # contact transitions, making the reference less responsive than the
        # (near-raw, causal-EMA) live teleop signal -> worse real-robot
        # following. Enable with --smooth only when explicitly desired.
        if do_smooth:
            qpos_list = smooth_motion_data(qpos_list, window_length=40, polyorder=3)
        qpos_list = np.array(qpos_list)

        # ---- Height fix on the SAME model used for deploy/training ----
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]
        dof_pos = qpos_list[:, 7:]

        mj_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        mj_data = mj.MjData(mj_model)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file_folder", required=True, type=str)
    parser.add_argument("--robot", choices=list(ROBOT_XML_DICT.keys()), default="unitree_g1")
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pns", action="store_true", help="process Noitom PNS-format BVH")
    parser.add_argument("--save_23dof", action="store_true", help="save 23 DoF (default 29)")
    parser.add_argument("--smooth", action="store_true",
                        help="apply savgol smoothing (off by default; raw "
                             "retargets track better on the real robot)")
    args = parser.parse_args()

    MOTION_FPS = 90 if args.pns else 120
    dof_flag = "23dof" if args.save_23dof else "29dof"

    os.makedirs(args.save_path, exist_ok=True)
    print(f"[blue]GMR model:[/blue] {ROBOT_XML_DICT[args.robot]}")

    total_bvh, skipped, tasks = 0, 0, []
    for root, _dirs, files in os.walk(args.bvh_file_folder):
        for f in files:
            if not f.endswith(".bvh"):
                continue
            total_bvh += 1
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, args.bvh_file_folder)
            filename = os.path.splitext(f)[0]
            output_dir = os.path.join(args.save_path, os.path.dirname(rel_path))
            output_path = os.path.join(output_dir, f"{filename}_{MOTION_FPS}Hz_{dof_flag}.npz")
            if os.path.exists(output_path) and not args.overwrite:
                skipped += 1
                continue
            tasks.append((full_path, rel_path, {
                "save_path": args.save_path,
                "robot": args.robot,
                "pns": args.pns,
                "motion_fps": MOTION_FPS,
                "save_23dof": args.save_23dof,
                "smooth": args.smooth,
            }))

    print(f"[green]Found {total_bvh} BVH files[/green]; to process: {len(tasks)}, skipped: {skipped}")
    if not tasks:
        print("[green]Nothing to do.[/green]")
        sys.exit(0)

    with Pool(processes=args.num_proc) as pool:
        results = list(tqdm(pool.imap(process_single_bvh, tasks), total=len(tasks)))

    success = sum(r["status"] == STATUS_SUCCESS for r in results)
    failed = [r for r in results if r["status"] == STATUS_ERROR]
    print("\n" + "=" * 60)
    print(f"[green]Success:[/green] {success}  [red]Failed:[/red] {len(failed)}  "
          f"[magenta]Skipped:[/magenta] {skipped}  Total: {total_bvh}")
    if failed:
        log_path = os.path.join(args.save_path, "failed_files.log")
        with open(log_path, "w") as fh:
            for r in failed:
                print(f"[red]FAIL[/red] {r['file_path']}: {r['error_msg']}")
                fh.write(f"File: {r['file_path']}\nError: {r['error_msg']}\n{'-'*40}\n")
        print(f"[yellow]Failed list -> {log_path}[/yellow]")
