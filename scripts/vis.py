import os
import time
import pickle
import imageio
import argparse
import numpy as np

import mujoco
import mujoco.viewer
from mujoco import Renderer

from tracking import constants as consts


# ==================== Constants ====================

ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Excluded joint indices for different qpos dimensions
SHAPE_MAP = {
    7 + 27: {13, 14},
    7 + 23: {13, 14, 20, 21, 27, 28},
    7 + 21: {13, 14, 19, 20, 21, 26, 27, 28},
}


# ==================== Utility Functions ====================

def get_qpos_ids(mj_model, names):
    """Get qpos indices for named joints in the MuJoCo model."""
    return np.hstack([mj_model.joint(n).qposadr for n in names])


def pad_qpos(qpos):
    """Pad qpos to full 36-dim (7 + 29 joints) if needed."""
    full_dim = 7 + len(ACTION_JOINT_NAMES)  # 36
    if qpos.shape[1] == full_dim:
        return qpos
    excluded = SHAPE_MAP.get(qpos.shape[1])
    if excluded is None:
        raise ValueError(
            f"Unsupported qpos dimension: {qpos.shape[1]}, "
            f"expected one of {[full_dim] + list(SHAPE_MAP.keys())}"
        )
    included = np.setdiff1d(np.arange(29), list(excluded))
    qpos_new = np.zeros((len(qpos), full_dim), dtype=qpos.dtype)
    qpos_id = np.hstack([np.arange(7), 7 + included])
    qpos_new[:, qpos_id] = qpos
    return qpos_new


def images_to_video(image_list, output_filename, fps=30):
    """Save a list of RGB images as a video file."""
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    frames = [np.clip(np.asarray(img), 0, 255).astype(np.uint8) for img in image_list]
    ext = os.path.splitext(output_filename)[1].lower()
    try:
        import imageio_ffmpeg  # noqa: F401
        kw = dict(fps=float(fps))
        if ext in {".mp4", ".m4v", ".mov"}:
            kw.update(codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])
        with imageio.get_writer(output_filename, format="FFMPEG", **kw) as w:
            for f in frames:
                w.append_data(f)
        return
    except Exception:
        pass
    with imageio.get_writer(output_filename, fps=float(fps)) as w:
        for f in frames:
            w.append_data(f)


# ==================== Core Visualization ====================

def visualize_qpos(
    qpos: np.ndarray,
    xml_path: str = str(consts.TRACK_XML),
    video_path: str = None,
    fps: int = 50,
    width: int = 640,
    height: int = 480
):
    """
    Render a qpos trajectory with MuJoCo and save as video in viewer.

    Args:
        qpos: (T, D) array, format [x, y, z, qw, qx, qy, qz, joint1, ...]
        xml_path: path to MuJoCo XML model file
        video_path: output video file path (None to skip saving)
        fps: frames per second for the output video
        width: video frame width in pixels
        height: video frame height in pixels
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    qpos = pad_qpos(qpos)

    mj_model = consts.load_mj_model(xml_path)
    mj_data = mujoco.MjData(mj_model)

    qpos_ids = get_qpos_ids(mj_model, ACTION_JOINT_NAMES)
    num_steps = len(qpos)
    dt = 1.0 / fps

    # ---- Interactive viewer ----
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 0
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance = 2.5

    # ---- Offscreen renderer for video ----
    renderer = None
    if video_path is not None:
        renderer = Renderer(mj_model, height=height, width=width)
        render_cam = mujoco.MjvCamera()
        render_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        render_cam.trackbodyid = 0
        render_cam.azimuth = 90.0
        render_cam.elevation = -20.0
        render_cam.distance = 2.5

    frames = []
    for t in range(num_steps):
        mj_data.qpos[:] = 0
        mj_data.qpos[:7] = qpos[t, :7]
        mj_data.qpos[qpos_ids] = qpos[t, 7:]
        mujoco.mj_forward(mj_model, mj_data)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(dt)

        if renderer is not None:
            renderer.update_scene(mj_data, render_cam)
            frames.append(renderer.render().copy())

    if viewer is not None and viewer.is_running():
        viewer.close()

    if video_path is not None and frames:
        images_to_video(frames, video_path, fps=fps)
        print(f"Video saved to: {video_path}  ({len(frames)} frames, {fps} fps, {len(frames) / fps:.1f}s)")


# ==================== Data Loading ====================

def load_qpos(path):
    """Load qpos array from .npz / .npy / .pkl file."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
    elif path.endswith(".npy"):
        data = np.load(path, allow_pickle=True).astype(np.float32)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    fps = data.get('frequency', 50)
    return fps, np.asarray(data["qpos"], dtype=np.float32)


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize qpos trajectory as video")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to qpos file (.npz / .npy / .pkl)")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output video path (default: None, set to save video)")
    parser.add_argument("--xml_path", type=str,
                        default=str(consts.TRACK_XML),
                        help="Path to MuJoCo XML model")
    parser.add_argument("--width", type=int, default=640,
                        help="Video width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height (default: 480)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start frame index (optional)")
    parser.add_argument("--end", type=int, default=None,
                        help="End frame index (optional)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed ratio (default: 1.0, e.g. 2.0 for 2x faster)")
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")

    fps, qpos = load_qpos(args.path)
    qpos = qpos[args.start:args.end]
    play_fps = fps * args.speed
    print(f"Loaded qpos: shape={qpos.shape}, frames={len(qpos)}, duration={len(qpos)/fps:.1f}s, "
          f"fps={fps}hz, speed={args.speed}x, play_fps={play_fps}hz")

    visualize_qpos(
        qpos,
        xml_path=args.xml_path,
        video_path=args.video_path,
        fps=play_fps,
        width=args.width,
        height=args.height
    )
