import os
import uuid
import signal
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import mujoco
import imageio
import numpy as np
import gradio as gr
from jax import tree_util as jtu
from loop_rate_limiters import RateLimiter

from tracking.policy import get_policy_onnx
from utils.ref_ghost import RefGhostRenderer
from scripts.inference import InferenceArgs, _convert_traj_to_kpt, _load_npz_with_qpos
from tracking.infer_utils import G1TrackInferFn, G1TrackMjSim, g1_infer_env_config, apply_ema_qpos
from tracking.metrics import (
    calculate_joint_tracking_error,
    calculate_kpt_mae_error,
    calculate_max_errors,
    calculate_root_tracking_error,
    calculate_trajectory_length,
)

DEFAULTS = InferenceArgs()
_GLOBAL_MJ_SIM = None
_GLOBAL_STATE = None
_GLOBAL_RENDERER = None  # Off-screen renderer for video capture
_GLOBAL_REF_GHOST = None  # Reference-pose ghost overlay
_CURRENT_RUN_ID = None
_CURRENT_FRICTION = 1.0  # Current floor friction coefficient
_ORIGINAL_PAIR_FRICTION = None  # Store original pair_friction for scaling
_ORIGINAL_GEOM_FRICTION = None  # Store original geom_friction for scaling
UI_UPDATE_EVERY = 5  # throttle UI updates to reduce flicker
DEFAULT_OUTPUT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
VIDEO_OUTPUT_DIR = Path("storage/videos")  # Directory for saving videos


def _set_floor_friction(mj_model: mujoco.MjModel, friction: float):
    """Set friction coefficient for all contact pairs.

    Args:
        mj_model: MuJoCo model
        friction: Friction scale (0.0 ~ 2.0, default 1.0)
            - 1.0 = original friction (no change)
            - 0.5 = half friction
            - 0.0 = zero friction (ice)

    Note:
        This scales all pair_friction values uniformly for demo/testing purposes.

        pair_friction has 5 components: [slide1, slide2, spin, roll1, roll2]
        - slide1, slide2: tangential friction in two orthogonal directions
        - spin: torsional friction around contact normal
        - roll1, roll2: rolling friction in two orthogonal directions

        Note: MuJoCo does NOT distinguish static vs dynamic friction.

        During training, domain_randomize.py uses absolute friction values
        for tangential components.
    """
    global _ORIGINAL_PAIR_FRICTION, _ORIGINAL_GEOM_FRICTION

    # Save original values on first call
    if _ORIGINAL_PAIR_FRICTION is None:
        _ORIGINAL_PAIR_FRICTION = mj_model.pair_friction.copy()
    if _ORIGINAL_GEOM_FRICTION is None:
        floor_geom_id = mj_model.geom("floor").id
        _ORIGINAL_GEOM_FRICTION = mj_model.geom_friction[floor_geom_id].copy()

    # Scale all pairs from original values (not cumulative)
    npair = mj_model.npair
    if npair > 0:
        mj_model.pair_friction[:, :] = _ORIGINAL_PAIR_FRICTION * friction

    # Also set geometry friction as fallback
    floor_geom_id = mj_model.geom("floor").id
    mj_model.geom_friction[floor_geom_id] = _ORIGINAL_GEOM_FRICTION * friction

    print(f"Friction scaled by {friction:.2f} for {npair} contact pairs")


def _get_renderer(mj_model: mujoco.MjModel):
    """Get or create a renderer for video capture.

    Uses the model's offscreen framebuffer dimensions defined in XML.
    """
    global _GLOBAL_RENDERER
    if _GLOBAL_RENDERER is None:
        # Use the model's offscreen framebuffer size
        width = mj_model.vis.global_.offwidth
        height = mj_model.vis.global_.offheight
        print(f"Creating renderer with size: {width}x{height}")
        _GLOBAL_RENDERER = mujoco.Renderer(mj_model, width=width, height=height)
    return _GLOBAL_RENDERER


def _capture_frame(
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ref_ghost: RefGhostRenderer | None = None,
) -> np.ndarray:
    """Capture a single frame from the simulation, optionally with ref ghost."""
    renderer.update_scene(mj_data, camera="track")
    if ref_ghost is not None:
        ref_ghost.add_to_scene(renderer.scene)
    return renderer.render()


def _get_ref_ghost(mj_model: mujoco.MjModel) -> RefGhostRenderer:
    """Get or create a singleton reference-pose ghost renderer."""
    global _GLOBAL_REF_GHOST
    if _GLOBAL_REF_GHOST is None or _GLOBAL_REF_GHOST.mj_model is not mj_model:
        _GLOBAL_REF_GHOST = RefGhostRenderer(mj_model)
    return _GLOBAL_REF_GHOST


def _save_video(frames: list, output_path: Path, fps: int = 50):
    """Save a list of frames as a video file."""
    if not frames:
        print("No frames to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Video saved to: {output_path}")


def _get_mj_sim(init_qpos: np.ndarray, friction: float = 1.0, ctrl_dt: float = 0.01):
    """Reuse a single viewer across runs and keep it centered on the robot.

    Args:
        init_qpos: Initial joint positions
        friction: Floor friction coefficient (0.0 ~ 2.0, default 1.0)
        ctrl_dt: Control timestep in seconds
    """
    global _GLOBAL_MJ_SIM, _GLOBAL_STATE, _CURRENT_FRICTION, _GLOBAL_RENDERER
    init_qpos = np.asarray(init_qpos, dtype=np.float32)

    if _GLOBAL_MJ_SIM is None:
        _GLOBAL_MJ_SIM = G1TrackMjSim(init_qpos=init_qpos, headless=False, ctrl_dt=ctrl_dt)
        _GLOBAL_STATE = _GLOBAL_MJ_SIM.init_state()
        if hasattr(_GLOBAL_MJ_SIM, "viewer") and _GLOBAL_MJ_SIM.viewer is not None:
            cam = _GLOBAL_MJ_SIM.viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 0
            cam.azimuth = 90.0
            cam.elevation = -20.0
            cam.distance = 2.0
    else:
        _GLOBAL_MJ_SIM.init_qpos = init_qpos
        if _GLOBAL_MJ_SIM.ctrl_dt != ctrl_dt:
            _GLOBAL_MJ_SIM.ctrl_dt = ctrl_dt
            _GLOBAL_MJ_SIM.num_sim_substeps = int(ctrl_dt / _GLOBAL_MJ_SIM.sim_dt)

    # Apply floor friction setting
    _set_floor_friction(_GLOBAL_MJ_SIM.mj_model, friction)
    _CURRENT_FRICTION = friction

    _GLOBAL_STATE = _GLOBAL_MJ_SIM.reset(_GLOBAL_STATE)
    return _GLOBAL_MJ_SIM, _GLOBAL_STATE


def parse_cli_defaults() -> argparse.Namespace:
    """Allow overriding the initial UI defaults via CLI without requiring any arguments."""
    parser = argparse.ArgumentParser(description="Gradio UI for Humanoid-GPT inference")
    parser.add_argument("--onnx_track", default=DEFAULTS.onnx_track)
    parser.add_argument("--privileged", action="store_true", default=DEFAULTS.privileged)
    parser.add_argument("--track_dir", default=DEFAULTS.track_dir)
    parser.add_argument("--freq", type=int, default=DEFAULTS.freq)
    return parser.parse_args()


def prepare_run_id():
    """Generate a new run id and cancel any ongoing run."""
    global _CURRENT_RUN_ID
    run_id = str(uuid.uuid4())
    _CURRENT_RUN_ID = run_id
    return run_id


def _prepare_traj_data(track_dir: str, convert_mj_model: mujoco.MjModel, freq_tgt: int):
    data_path = Path(track_dir)
    if data_path.is_file():
        traj_files = [data_path]
    elif data_path.is_dir():
        traj_files = sorted(list(data_path.glob("*.npz")))
    else:
        raise ValueError(f"{data_path} not exist.")

    if not traj_files:
        raise ValueError(f"No .npz reference trajectories found under {data_path}.")

    traj_data = []
    for file in traj_files:
        raw_data = _load_npz_with_qpos(file)
        # Apply EMA smoothing to reference qpos before conversion
        raw_data["qpos"] = apply_ema_qpos(raw_data["qpos"])
        traj_data.append(_convert_traj_to_kpt(raw_data, convert_mj_model, freq_tgt))
    return traj_data, data_path, traj_files


def _boot_viewer_at_start(defaults: argparse.Namespace):
    """Open viewer once at startup with default mocap to show initial pose."""
    try:
        merged = {**DEFAULTS.__dict__, **vars(defaults)}
        args = InferenceArgs(**merged)
        convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        traj_data, _, _ = _prepare_traj_data(args.track_dir, convert_mj_model, args.freq)
        _init_qpos = traj_data[0]["qpos"][0].copy()
        _init_qpos[:2] = 0.0
        _get_mj_sim(_init_qpos, ctrl_dt=1 / args.freq)
    except Exception:
        traceback.print_exc()


def run_inference_gradio(
        run_id,
        onnx_track,
        privileged,
        track_dir,
        freq,
        floor_friction,
        save_video,
        show_ref_ghost,
        progress: gr.Progress | None = None,
):
    """Generator for Gradio that streams metrics and a progress bar."""
    try:
        global _CURRENT_RUN_ID, _GLOBAL_RENDERER
        if not run_id:
            run_id = str(uuid.uuid4())
        _CURRENT_RUN_ID = run_id

        # Validate floor friction
        floor_friction = float(floor_friction) if floor_friction is not None else 1.0
        floor_friction = max(0.0, min(2.0, floor_friction))  # Clamp to [0.0, 2.0]

        # Video saving flag
        save_video = bool(save_video)

        # Reference-pose ghost overlay flag
        show_ref_ghost = bool(show_ref_ghost)

        freq = int(freq) if freq else DEFAULTS.freq
        args = InferenceArgs(
            onnx_track=onnx_track or DEFAULTS.onnx_track,
            privileged=bool(privileged),
            track_dir=track_dir or DEFAULTS.track_dir,
            freq=freq,
            headless=False,
        )

        def r4(x):
            return float(f"{float(x):.4g}")

        env_cfg = g1_infer_env_config(ctrl_dt=1 / freq)

        # Check again before loading data
        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        traj_data, data_path, traj_files = _prepare_traj_data(args.track_dir, convert_mj_model, args.freq)
        total_steps = sum(len(traj["qpos"]) for traj in traj_data)
        if total_steps == 0:
            raise ValueError("Reference trajectories are empty.")

        # Check again before initializing simulation
        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        _init_qpos = traj_data[0]["qpos"][0].copy()
        _init_qpos[:2] = 0.0
        mj_sim, state = _get_mj_sim(_init_qpos, friction=floor_friction, ctrl_dt=1 / args.freq)
        ctrl_rate = RateLimiter(frequency=args.freq, warn=False)

        ref_ghost = _get_ref_ghost(mj_sim.mj_model) if show_ref_ghost else None

        # Initialize renderer for video capture if needed
        renderer = None
        if save_video:
            renderer = _get_renderer(mj_sim.mj_model)

        # Check again before loading model
        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        if not args.onnx_track.endswith(".onnx"):
            raise ValueError(f"Unsupported onnx_track format: {args.onnx_track} (expected .onnx)")
        policy = get_policy_onnx(args)

        # Final check before starting inference
        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, policy, privileged=args.privileged)

        steps_done = 0
        latest_length_ratio = 0.0

        for traj_id in range(len(traj_data)):
            if run_id != _CURRENT_RUN_ID:
                yield DEFAULT_OUTPUT
                return
            ref_traj = traj_data[traj_id]
            traj_len = len(ref_traj["qpos"])
            ref_start_xy = ref_traj["qpos"][0, :2].copy()

            _init_qpos = ref_traj["qpos"][0].copy()
            _init_qpos[:2] = 0.0
            mj_sim.init_qpos[:] = _init_qpos
            state = mj_sim.reset(state)

            if "qvel" in ref_traj:
                _init_qvel = ref_traj["qvel"][0]
                state.mj_data.qpos[:] = _init_qpos
                state.mj_data.qvel[:] = _init_qvel
                mujoco.mj_forward(mj_sim.mj_model, state.mj_data)
            if data_path.is_file():
                file_name = data_path.name
            else:
                file_name = traj_files[traj_id].name if traj_id < len(traj_files) else f"traj_{traj_id}.npz"

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

            # Initialize frame list for video capture
            video_frames = [] if save_video else None

            for track_step in range(traj_len):
                # Check cancellation at the start of each step
                if run_id != _CURRENT_RUN_ID:
                    yield DEFAULT_OUTPUT
                    return
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

                if ref_ghost is not None:
                    ghost_qpos = ref_curr["qpos"][0].copy()
                    ghost_qpos[:2] -= ref_start_xy
                    ref_ghost.set_qpos(ghost_qpos)
                    viewer = getattr(mj_sim, "viewer", None)
                    if viewer is not None:
                        ref_ghost.reset_scene(viewer.user_scn)
                        ref_ghost.add_to_scene(viewer.user_scn)
                else:
                    # Clear any lingering ghost geoms from a previous run.
                    viewer = getattr(mj_sim, "viewer", None)
                    if viewer is not None:
                        viewer.user_scn.ngeom = 0

                mj_sim.view(state)

                # Capture frame for video if enabled
                if save_video and renderer is not None:
                    frame = _capture_frame(renderer, mj_sim.mj_model, state.mj_data, ref_ghost)
                    video_frames.append(frame)

                # Check cancellation before sleep (to respond faster)
                if run_id != _CURRENT_RUN_ID:
                    yield DEFAULT_OUTPUT
                    return

                ctrl_rate.sleep()

                # Check cancellation after sleep as well
                if run_id != _CURRENT_RUN_ID:
                    yield DEFAULT_OUTPUT
                    return

                steps_done += 1
                # Update running completion ratio within the current trajectory
                latest_length_ratio = r4((track_step + 1) / traj_len)
                should_update = (
                        steps_done % UI_UPDATE_EVERY == 0
                        or (traj_id == len(traj_data) - 1 and track_step == traj_len - 1)
                )
                if should_update:
                    percent = r4((steps_done / total_steps) * 100)
                    # Calculate max errors across all frames so far
                    max_errors = calculate_max_errors(traj_metrics)
                    result = (
                        percent,
                        r4(latest_length_ratio),
                        r4(kpt_pos_mae),
                        r4(kpt_rot_mae),
                        r4(joint_pos_mae),
                        r4(joint_vel_mae),
                        r4(root_pos_err_mm),
                        r4(root_vel_err_mms),
                        r4(root_yaw_err),
                        r4(max_errors["max_kpt_pos_error"]),
                        r4(max_errors["max_kpt_rot_error"]),
                        r4(max_errors["max_joint_pos_error"]),
                        r4(max_errors["max_joint_vel_error"]),
                        r4(max_errors["max_root_pos_error"]),
                        r4(max_errors["max_root_vel_error"]),
                        r4(max_errors["max_root_yaw_error"]),
                    )
                    yield result
                    # Check cancellation after each yield to respond immediately
                    if run_id != _CURRENT_RUN_ID:
                        return

            # Check cancellation before finalizing trajectory metrics
            if run_id != _CURRENT_RUN_ID:
                yield DEFAULT_OUTPUT
                return

            # finalize metrics for this trajectory
            actual_trajectory_length = len(ref_traj["qpos"])
            latest_length_ratio, termination_step = calculate_trajectory_length(
                traj_metrics["state_history"], ref_traj, mj_sim.mj_model
            )
            latest_length_ratio = r4(latest_length_ratio)
            avg_kpt_pos_error = r4(np.mean(traj_metrics["kpt_pos_errors"]))
            avg_kpt_rot_error = r4(np.mean(traj_metrics["kpt_rot_errors"]))
            avg_joint_pos_error = r4(np.mean(traj_metrics["joint_pos_errors"]))
            avg_joint_vel_error = r4(np.mean(traj_metrics["joint_vel_errors"]))
            avg_root_pos_error = r4(np.mean(traj_metrics["root_pos_errors"]))
            avg_root_vel_error = r4(np.mean(traj_metrics["root_vel_errors"]))
            avg_root_yaw_error = r4(np.mean(traj_metrics["root_yaw_errors"]))

            # Calculate max errors across all frames
            max_errors = calculate_max_errors(traj_metrics)

            # Print metrics for current trajectory
            print("\n" + "=" * 80)
            print(f"Trajectory {traj_id} completed ({file_name}):")
            print("=" * 80)
            print(f"  Completion: {latest_length_ratio:.4f} ({termination_step}/{actual_trajectory_length} steps)")
            print(f"  KPT Position MAE: {avg_kpt_pos_error:.6f} m (Max: {max_errors['max_kpt_pos_error']:.6f} m)")
            print(f"  KPT Rotation MAE: {avg_kpt_rot_error:.6f} rad (Max: {max_errors['max_kpt_rot_error']:.6f} rad)")
            print(
                f"  Joint Position MAE: {avg_joint_pos_error:.6f} rad (Max: {max_errors['max_joint_pos_error']:.6f} rad)")
            print(
                f"  Joint Velocity MAE: {avg_joint_vel_error:.6f} rad/s (Max: {max_errors['max_joint_vel_error']:.6f} rad/s)")
            print(f"  Root Pos Error: {avg_root_pos_error:.3f} mm (Max: {max_errors['max_root_pos_error']:.3f} mm)")
            print(f"  Root Vel Error: {avg_root_vel_error:.3f} mm/s (Max: {max_errors['max_root_vel_error']:.3f} mm/s)")
            print(f"  Root Yaw Error: {avg_root_yaw_error:.6f} rad (Max: {max_errors['max_root_yaw_error']:.6f} rad)")
            print("=" * 80 + "\n")

            # Save video if enabled
            if save_video and video_frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_name = file_name.replace(".npz", "")
                video_path = VIDEO_OUTPUT_DIR / f"{video_name}_{timestamp}.mp4"
                _save_video(video_frames, video_path, fps=int(args.freq))

            yield (
                r4((steps_done / total_steps) * 100),
                latest_length_ratio,
                avg_kpt_pos_error,
                avg_kpt_rot_error,
                avg_joint_pos_error,
                avg_joint_vel_error,
                avg_root_pos_error,
                avg_root_vel_error,
                avg_root_yaw_error,
                r4(max_errors["max_kpt_pos_error"]),
                r4(max_errors["max_kpt_rot_error"]),
                r4(max_errors["max_joint_pos_error"]),
                r4(max_errors["max_joint_vel_error"]),
                r4(max_errors["max_root_pos_error"]),
                r4(max_errors["max_root_vel_error"]),
                r4(max_errors["max_root_yaw_error"]),
            )
            # Check cancellation after final trajectory yield
            if run_id != _CURRENT_RUN_ID:
                return

    except Exception as exc:  # pragma: no cover - user feedback path
        traceback.print_exc()
        # Yield numeric values (0.0) instead of error messages for Number components
        yield DEFAULT_OUTPUT


def build_interface(defaults: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="Humanoid-GPT Inference") as demo:
        gr.Markdown("# Humanoid-GPT Gradio Inference\nLive metrics and viewer.")
        run_id_state = gr.State("")
        with gr.Row():
            onnx_track_box = gr.Textbox(label="onnx_track", value=defaults.onnx_track)
        with gr.Row():
            privileged_box = gr.Checkbox(label="privileged", value=defaults.privileged)
            track_dir_box = gr.Textbox(label="track_dir", value=defaults.track_dir)
            freq_box = gr.Number(label="freq (Hz)", value=defaults.freq, precision=0)

        with gr.Row():
            floor_friction_slider = gr.Slider(
                label="Floor Friction",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="Ground friction coefficient (0.0=ice, 1.0=normal, 2.0=high friction)"
            )
            save_video_checkbox = gr.Checkbox(
                label="video",
                value=False,
                info="Save video at the end of each trajectory"
            )
            show_ref_ghost_checkbox = gr.Checkbox(
                label="ref ghost",
                value=True,
                info="Overlay reference motion as a red translucent ghost"
            )

        run_btn = gr.Button("Start Inference")

        progress_bar = gr.Slider(label="Progress (%)", minimum=0, maximum=100, value=0, step=0.0001, interactive=False)
        traj_ratio = gr.Number(label="traj_length_ratio", precision=4)

        # Mean errors column
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Mean Errors")
                kpt_pos_error = gr.Number(label="kpt_pos_error", precision=4)
                kpt_rot_error = gr.Number(label="kpt_rot_error", precision=4)
                joint_pos_error = gr.Number(label="joint_pos_error", precision=4)
                joint_vel_error = gr.Number(label="joint_vel_error", precision=4)
                root_pos_error = gr.Number(label="root_pos_error", precision=4)
                root_vel_error = gr.Number(label="root_vel_error", precision=4)
                root_yaw_error = gr.Number(label="root_yaw_error", precision=4)

            # Max errors column
            with gr.Column():
                gr.Markdown("### Max Errors")
                max_kpt_pos_error = gr.Number(label="max_kpt_pos_error", precision=4)
                max_kpt_rot_error = gr.Number(label="max_kpt_rot_error", precision=4)
                max_joint_pos_error = gr.Number(label="max_joint_pos_error", precision=4)
                max_joint_vel_error = gr.Number(label="max_joint_vel_error", precision=4)
                max_root_pos_error = gr.Number(label="max_root_pos_error", precision=4)
                max_root_vel_error = gr.Number(label="max_root_vel_error", precision=4)
                max_root_yaw_error = gr.Number(label="max_root_yaw_error", precision=4)

        run_btn.click(
            prepare_run_id,
            outputs=run_id_state,
            queue=False,
        ).then(
            run_inference_gradio,
            inputs=[run_id_state, onnx_track_box, privileged_box, track_dir_box, freq_box,
                    floor_friction_slider, save_video_checkbox, show_ref_ghost_checkbox],
            outputs=[
                progress_bar,
                traj_ratio,
                kpt_pos_error,
                kpt_rot_error,
                joint_pos_error,
                joint_vel_error,
                root_pos_error,
                root_vel_error,
                root_yaw_error,
                max_kpt_pos_error,
                max_kpt_rot_error,
                max_joint_pos_error,
                max_joint_vel_error,
                max_root_pos_error,
                max_root_vel_error,
                max_root_yaw_error,
            ],
            queue=True,
        )
    return demo


def _enable_queue(app: gr.Blocks, concurrency: int = 1, max_size: int | None = None):
    """Enable Gradio queue with best-effort compatibility across versions.

    Using concurrency=1 ensures that tasks run sequentially, allowing immediate
    cancellation of the old task when a new one starts.
    """
    import inspect

    kwargs = {}
    sig = inspect.signature(app.queue)
    if "concurrency_count" in sig.parameters:
        kwargs["concurrency_count"] = concurrency
    if "max_size" in sig.parameters and max_size is not None:
        kwargs["max_size"] = max_size
    # Call with discovered kwargs (or none if unsupported)
    app.queue(**kwargs)


if __name__ == "__main__":
    # Handle Ctrl+C: force exit immediately
    def signal_handler(sig, frame):
        print("\nReceived Ctrl+C, forcing exit...")
        os._exit(1)


    signal.signal(signal.SIGINT, signal_handler)

    defaults = parse_cli_defaults()
    _boot_viewer_at_start(defaults)
    app = build_interface(defaults)
    _enable_queue(app, concurrency=1)  # Use concurrency=1 for immediate task cancellation
    app.launch()
