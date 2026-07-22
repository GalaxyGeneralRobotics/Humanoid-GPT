"""
Parallel headless video recording from npz motion files.

Usage:
    python -m utils.record_video_parallel --src_dir /path/to/npz_folder --save_dir /path/to/video_output --num_workers 8

Speed notes:
    * The MuJoCo model and the (EGL/GL) Renderer are created ONCE per worker
      process and then reused across every file it handles. Recreating the GL
      context / recompiling the model for every single npz is what made the old
      version slow (that overhead dwarfs the actual rendering for short clips).
    * Only kinematics are computed per frame (mj_kinematics + mj_comPos) instead
      of a full mj_forward, since rendering does not need collisions/dynamics.
    * Videos are encoded with the ultrafast x264 preset by default.
"""

import os
import sys

# Set environment variables BEFORE importing mujoco (for main process)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import tyro
import numpy as np
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

from tracking import constants as consts
from utils.video_utils import images_to_video


@dataclass
class Cfgs:
    src_dir: str
    save_dir: str
    num_workers: int = 8
    output_fps: int = 30  # Output video framerate (subsample from source)
    video_width: int = 640
    video_height: int = 480
    xml_path: str = str(consts.DEBUG_TRACK_XML)
    overwrite: bool = False  # Re-render even if the output video already exists
    preset: str = "ultrafast"  # x264 preset; "ultrafast" encodes much faster
    # Spread EGL rendering across this many GPUs (round-robin by worker index).
    # 0 = leave device selection to the driver (single-GPU default behaviour).
    num_gpus: int = 0
    # Encoder threads per clip. Keep small (1) when running many workers, since
    # parallelism already comes from the worker pool -- otherwise every ffmpeg
    # process grabs all cores and they thrash each other.
    ffmpeg_threads: int = 1


# Per-worker persistent state, populated once by ``_init_worker``.
_WORKER: dict = {}


def _init_worker(xml_path: str, video_width: int, video_height: int, num_gpus: int = 0):
    """Initialize the (expensive) MuJoCo model + renderer once per worker."""
    import os
    import sys
    import multiprocessing as mp

    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"

    # Round-robin this worker onto one of the GPUs so EGL rendering does not all
    # pile onto device 0. Must be set BEFORE the renderer creates its context.
    if num_gpus and num_gpus > 0:
        ident = mp.current_process()._identity
        widx = (ident[0] - 1) if ident else 0
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(widx % num_gpus)

    import mujoco
    from mujoco import Renderer

    mj_model = consts.load_mj_model(xml_path)
    mj_data = mujoco.MjData(mj_model)

    renderer = Renderer(mj_model, height=video_height, width=video_width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.azimuth = 90.0
    cam.elevation = -20.0
    cam.distance = 2.5

    _WORKER["mujoco"] = mujoco
    _WORKER["model"] = mj_model
    _WORKER["data"] = mj_data
    _WORKER["renderer"] = renderer
    _WORKER["cam"] = cam


def _worker(npz_path_str: str, video_path_str: str, output_fps: int, preset: str, ffmpeg_threads: int = 1) -> bool:
    """Render a single npz file to video, reusing the per-worker resources."""
    mujoco = _WORKER["mujoco"]
    mj_model = _WORKER["model"]
    mj_data = _WORKER["data"]
    renderer = _WORKER["renderer"]
    cam = _WORKER["cam"]

    try:
        data = np.load(npz_path_str, allow_pickle=True)

        if "qpos" not in data:
            print(f"[WARN] No 'qpos' found in {npz_path_str}, skipping.")
            return False

        qpos = np.asarray(data["qpos"], dtype=np.float32)
        num_steps = len(qpos)

        src_fps = int(data["frequency"]) if "frequency" in data else 50

        if num_steps < 10:
            print(f"[WARN] Too few frames ({num_steps}) in {npz_path_str}, skipping.")
            return False

        actual_output_fps = min(output_fps, src_fps)
        frame_step = max(1, src_fps // actual_output_fps)
        actual_output_fps = src_fps // frame_step

        nq = mj_model.nq

        frames = []
        for t in range(0, num_steps, frame_step):
            q = qpos[t]
            n = min(len(q), nq)
            mj_data.qpos[:n] = q[:n]
            # Rendering only needs kinematics (+ com for the tracking camera),
            # not the full dynamics/collision pipeline of mj_forward.
            mujoco.mj_kinematics(mj_model, mj_data)
            mujoco.mj_comPos(mj_model, mj_data)

            renderer.update_scene(mj_data, cam)
            frames.append(renderer.render())

        Path(video_path_str).parent.mkdir(parents=True, exist_ok=True)
        images_to_video(
            frames, video_path_str, fps=actual_output_fps, color_format="RGB",
            preset=preset, threads=ffmpeg_threads,
        )
        print(
            f"[OK] Saved video: {video_path_str} "
            f"({len(frames)} frames @ {actual_output_fps}Hz, src: {num_steps} frames @ {src_fps}Hz)"
        )
        return True

    except Exception as e:
        print(f"[ERROR] Failed to process {npz_path_str}: {e}")
        return False


def main(args: Cfgs):
    print(f"Configuration: {args}")

    src_dir = Path(args.src_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find all npz files
    all_files = sorted(src_dir.rglob("*.npz"))
    if not all_files:
        print(f"No npz files found in {src_dir}")
        return

    print(f"Found {len(all_files)} npz files.")

    # Build jobs, resolving output paths up-front and skipping existing ones.
    jobs = []
    skipped = 0
    for file_path in all_files:
        rel_path = file_path.relative_to(src_dir)
        video_name = str(rel_path).replace("/", "_").replace("\\", "_")
        video_name = video_name.rsplit(".", 1)[0] + ".mp4"
        video_path = save_dir / video_name

        if not args.overwrite and video_path.exists():
            skipped += 1
            continue

        jobs.append((str(file_path), str(video_path), args.output_fps, args.preset, args.ffmpeg_threads))

    if skipped:
        print(f"Skipping {skipped} files that already have videos (use --overwrite to force).")
    if not jobs:
        print("Nothing to do.")
        return

    print(f"Processing {len(jobs)} files with {args.num_workers} workers (num_gpus={args.num_gpus})...")

    # Multi-process parallel execution. Each worker sets up its model + renderer
    # exactly once via the initializer, then reuses them across its files.
    ctx = mp.get_context("spawn")  # More robust for MuJoCo / GPU contexts
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.xml_path, args.video_width, args.video_height, args.num_gpus),
    ) as pool:
        results = pool.starmap(_worker, jobs, chunksize=1)

    ok = sum(1 for r in results if r)
    print(f"Done! {ok}/{len(jobs)} videos rendered. Saved to: {save_dir}")


if __name__ == "__main__":
    main(tyro.cli(Cfgs))
