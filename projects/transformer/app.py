"""Gradio UI for Transformer tracker inference.

Mirrors ``scripts/app.py`` but uses ``G1TrackTransformerInferFn`` so the
ONNX policy is driven by a rolling K-frame observation history.

Usage::

    python -m projects.transformer.app \\
        --onnx_track storage/ckpt/transformer.onnx \\
        --track_dir storage/test \\
        --history_len 4
"""

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
from projects.transformer.inference import InferenceArgs, _convert_traj_to_kpt, _load_npz_with_qpos
from projects.transformer.infer_utils import G1TrackTransformerInferFn
from tracking.infer_utils import G1TrackMjSim, g1_infer_env_config, apply_ema_qpos
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
_GLOBAL_RENDERER = None
_GLOBAL_REF_GHOST = None
_CURRENT_RUN_ID = None
_CURRENT_FRICTION = 1.0
_ORIGINAL_PAIR_FRICTION = None
_ORIGINAL_GEOM_FRICTION = None
UI_UPDATE_EVERY = 5
DEFAULT_OUTPUT = (0.0,) * 16
VIDEO_OUTPUT_DIR = Path("storage/videos")


def _set_floor_friction(mj_model: mujoco.MjModel, friction: float):
    global _ORIGINAL_PAIR_FRICTION, _ORIGINAL_GEOM_FRICTION
    if _ORIGINAL_PAIR_FRICTION is None:
        _ORIGINAL_PAIR_FRICTION = mj_model.pair_friction.copy()
    if _ORIGINAL_GEOM_FRICTION is None:
        floor_geom_id = mj_model.geom("floor").id
        _ORIGINAL_GEOM_FRICTION = mj_model.geom_friction[floor_geom_id].copy()
    if mj_model.npair > 0:
        mj_model.pair_friction[:, :] = _ORIGINAL_PAIR_FRICTION * friction
    floor_geom_id = mj_model.geom("floor").id
    mj_model.geom_friction[floor_geom_id] = _ORIGINAL_GEOM_FRICTION * friction


def _get_renderer(mj_model: mujoco.MjModel):
    global _GLOBAL_RENDERER
    if _GLOBAL_RENDERER is None:
        w = mj_model.vis.global_.offwidth
        h = mj_model.vis.global_.offheight
        _GLOBAL_RENDERER = mujoco.Renderer(mj_model, width=w, height=h)
    return _GLOBAL_RENDERER


def _capture_frame(
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ref_ghost: RefGhostRenderer | None = None,
) -> np.ndarray:
    renderer.update_scene(mj_data, camera="track")
    if ref_ghost is not None:
        ref_ghost.add_to_scene(renderer.scene)
    return renderer.render()


def _get_ref_ghost(mj_model: mujoco.MjModel) -> RefGhostRenderer:
    global _GLOBAL_REF_GHOST
    if _GLOBAL_REF_GHOST is None or _GLOBAL_REF_GHOST.mj_model is not mj_model:
        _GLOBAL_REF_GHOST = RefGhostRenderer(mj_model)
    return _GLOBAL_REF_GHOST


def _save_video(frames, output_path: Path, fps: int = 50):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Video saved to: {output_path}")


def _get_mj_sim(init_qpos, friction: float = 1.0, ctrl_dt: float = 0.02):
    global _GLOBAL_MJ_SIM, _GLOBAL_STATE, _CURRENT_FRICTION
    init_qpos = np.asarray(init_qpos, dtype=np.float32)
    if _GLOBAL_MJ_SIM is None:
        _GLOBAL_MJ_SIM = G1TrackMjSim(init_qpos=init_qpos, headless=False, ctrl_dt=ctrl_dt)
        _GLOBAL_STATE = _GLOBAL_MJ_SIM.init_state()
        if hasattr(_GLOBAL_MJ_SIM, "viewer") and _GLOBAL_MJ_SIM.viewer is not None:
            cam = _GLOBAL_MJ_SIM.viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 0
            cam.azimuth, cam.elevation, cam.distance = 90.0, -20.0, 2.0
    else:
        _GLOBAL_MJ_SIM.init_qpos = init_qpos
        if _GLOBAL_MJ_SIM.ctrl_dt != ctrl_dt:
            _GLOBAL_MJ_SIM.ctrl_dt = ctrl_dt
            _GLOBAL_MJ_SIM.num_sim_substeps = int(ctrl_dt / _GLOBAL_MJ_SIM.sim_dt)

    _set_floor_friction(_GLOBAL_MJ_SIM.mj_model, friction)
    _CURRENT_FRICTION = friction
    _GLOBAL_STATE = _GLOBAL_MJ_SIM.reset(_GLOBAL_STATE)
    return _GLOBAL_MJ_SIM, _GLOBAL_STATE


def _prepare_traj_data(track_dir, convert_mj_model, freq_tgt):
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
        raw = _load_npz_with_qpos(file)
        raw["qpos"] = apply_ema_qpos(raw["qpos"])
        traj_data.append(_convert_traj_to_kpt(raw, convert_mj_model, freq_tgt))
    return traj_data, data_path, traj_files


def parse_cli_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio UI for Transformer tracker")
    parser.add_argument("--onnx_track", default=DEFAULTS.onnx_track)
    parser.add_argument("--privileged", action="store_true", default=DEFAULTS.privileged)
    parser.add_argument("--track_dir", default=DEFAULTS.track_dir)
    parser.add_argument("--freq", type=int, default=DEFAULTS.freq)
    parser.add_argument("--history_len", type=int, default=DEFAULTS.history_len)
    return parser.parse_args()


def prepare_run_id():
    global _CURRENT_RUN_ID
    run_id = str(uuid.uuid4())
    _CURRENT_RUN_ID = run_id
    return run_id


def _boot_viewer_at_start(defaults: argparse.Namespace):
    try:
        merged = {**DEFAULTS.__dict__, **vars(defaults)}
        args = InferenceArgs(**merged)
        convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        traj_data, _, _ = _prepare_traj_data(args.track_dir, convert_mj_model, args.freq)
        iq = traj_data[0]["qpos"][0].copy()
        iq[:2] = 0.0
        _get_mj_sim(iq, ctrl_dt=1 / args.freq)
    except Exception:
        traceback.print_exc()


def run_inference_gradio(
    run_id,
    onnx_track,
    privileged,
    track_dir,
    freq,
    history_len,
    floor_friction,
    save_video,
    show_ref_ghost,
    progress: gr.Progress | None = None,
):
    try:
        global _CURRENT_RUN_ID
        if not run_id:
            run_id = str(uuid.uuid4())
        _CURRENT_RUN_ID = run_id

        floor_friction = max(0.0, min(2.0, float(floor_friction or 1.0)))
        save_video = bool(save_video)
        show_ref_ghost = bool(show_ref_ghost)
        freq = int(freq) if freq else DEFAULTS.freq
        history_len = max(1, int(history_len or DEFAULTS.history_len))

        args = InferenceArgs(
            onnx_track=onnx_track or DEFAULTS.onnx_track,
            privileged=bool(privileged),
            track_dir=track_dir or DEFAULTS.track_dir,
            freq=freq,
            history_len=history_len,
            headless=False,
        )

        def r4(x):
            return float(f"{float(x):.4g}")

        env_cfg = g1_infer_env_config(ctrl_dt=1 / freq)

        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        convert_mj_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        traj_data, data_path, traj_files = _prepare_traj_data(args.track_dir, convert_mj_model, args.freq)
        total_steps = sum(len(t["qpos"]) for t in traj_data)
        if total_steps == 0:
            raise ValueError("Reference trajectories are empty.")

        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        _init_qpos = traj_data[0]["qpos"][0].copy()
        _init_qpos[:2] = 0.0
        mj_sim, state = _get_mj_sim(_init_qpos, friction=floor_friction, ctrl_dt=1 / args.freq)
        ctrl_rate = RateLimiter(frequency=args.freq, warn=False)

        ref_ghost = _get_ref_ghost(mj_sim.mj_model) if show_ref_ghost else None
        renderer = _get_renderer(mj_sim.mj_model) if save_video else None

        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        if not args.onnx_track.endswith(".onnx"):
            raise ValueError(f"Unsupported onnx_track format: {args.onnx_track} (expected .onnx)")
        policy = get_policy_onnx(args)

        if run_id != _CURRENT_RUN_ID:
            yield DEFAULT_OUTPUT
            return

        infer_fn = G1TrackTransformerInferFn(
            env_cfg, mj_sim.mj_model, policy,
            privileged=args.privileged,
            history_len=args.history_len,
        )

        steps_done = 0
        latest_length_ratio = 0.0

        for traj_id in range(len(traj_data)):
            if run_id != _CURRENT_RUN_ID:
                yield DEFAULT_OUTPUT
                return
            ref_traj = traj_data[traj_id]
            traj_len = len(ref_traj["qpos"])
            ref_start_xy = ref_traj["qpos"][0, :2].copy()

            _iq = ref_traj["qpos"][0].copy()
            _iq[:2] = 0.0
            mj_sim.init_qpos[:] = _iq
            state = mj_sim.reset(state)
            infer_fn.reset_history()

            if "qvel" in ref_traj:
                state.mj_data.qpos[:] = _iq
                state.mj_data.qvel[:] = ref_traj["qvel"][0]
                mujoco.mj_forward(mj_sim.mj_model, state.mj_data)
            if data_path.is_file():
                file_name = data_path.name
            else:
                file_name = traj_files[traj_id].name if traj_id < len(traj_files) else f"traj_{traj_id}.npz"

            traj_metrics = {k: [] for k in (
                "kpt_pos_errors", "kpt_rot_errors",
                "joint_pos_errors", "joint_vel_errors",
                "root_pos_errors", "root_vel_errors", "root_yaw_errors",
                "state_history",
            )}
            video_frames = [] if save_video else None

            for track_step in range(traj_len):
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
                traj_metrics["state_history"].append({
                    "qpos": state.mj_data.qpos.copy(),
                    "qvel": state.mj_data.qvel.copy(),
                    "xpos": state.mj_data.xpos.copy(),
                    "xmat": state.mj_data.xmat.copy(),
                })

                if ref_ghost is not None:
                    ghost_qpos = ref_curr["qpos"][0].copy()
                    ghost_qpos[:2] -= ref_start_xy
                    ref_ghost.set_qpos(ghost_qpos)
                    viewer = getattr(mj_sim, "viewer", None)
                    if viewer is not None:
                        ref_ghost.reset_scene(viewer.user_scn)
                        ref_ghost.add_to_scene(viewer.user_scn)
                else:
                    viewer = getattr(mj_sim, "viewer", None)
                    if viewer is not None:
                        viewer.user_scn.ngeom = 0

                mj_sim.view(state)

                if save_video and renderer is not None:
                    video_frames.append(
                        _capture_frame(renderer, mj_sim.mj_model, state.mj_data, ref_ghost)
                    )

                if run_id != _CURRENT_RUN_ID:
                    yield DEFAULT_OUTPUT
                    return
                ctrl_rate.sleep()
                if run_id != _CURRENT_RUN_ID:
                    yield DEFAULT_OUTPUT
                    return

                steps_done += 1
                latest_length_ratio = r4((track_step + 1) / traj_len)
                should_update = (
                    steps_done % UI_UPDATE_EVERY == 0
                    or (traj_id == len(traj_data) - 1 and track_step == traj_len - 1)
                )
                if should_update:
                    me = calculate_max_errors(traj_metrics)
                    yield (
                        r4((steps_done / total_steps) * 100),
                        latest_length_ratio,
                        r4(kpt_pos_mae), r4(kpt_rot_mae),
                        r4(joint_pos_mae), r4(joint_vel_mae),
                        r4(root_pos_err_mm), r4(root_vel_err_mms), r4(root_yaw_err),
                        r4(me["max_kpt_pos_error"]), r4(me["max_kpt_rot_error"]),
                        r4(me["max_joint_pos_error"]), r4(me["max_joint_vel_error"]),
                        r4(me["max_root_pos_error"]), r4(me["max_root_vel_error"]),
                        r4(me["max_root_yaw_error"]),
                    )
                    if run_id != _CURRENT_RUN_ID:
                        return

            if run_id != _CURRENT_RUN_ID:
                yield DEFAULT_OUTPUT
                return

            actual_trajectory_length = len(ref_traj["qpos"])
            latest_length_ratio, termination_step = calculate_trajectory_length(
                traj_metrics["state_history"], ref_traj, mj_sim.mj_model
            )
            latest_length_ratio = r4(latest_length_ratio)
            avg_kpt_pos = r4(np.mean(traj_metrics["kpt_pos_errors"]))
            avg_kpt_rot = r4(np.mean(traj_metrics["kpt_rot_errors"]))
            avg_joint_pos = r4(np.mean(traj_metrics["joint_pos_errors"]))
            avg_joint_vel = r4(np.mean(traj_metrics["joint_vel_errors"]))
            avg_root_pos = r4(np.mean(traj_metrics["root_pos_errors"]))
            avg_root_vel = r4(np.mean(traj_metrics["root_vel_errors"]))
            avg_root_yaw = r4(np.mean(traj_metrics["root_yaw_errors"]))
            me = calculate_max_errors(traj_metrics)

            print("\n" + "=" * 80)
            print(f"Trajectory {traj_id} completed ({file_name}):")
            print("=" * 80)
            print(f"  Completion: {latest_length_ratio:.4f} ({termination_step}/{actual_trajectory_length} steps)")
            print(f"  KPT Position MAE: {avg_kpt_pos:.6f} m (Max: {me['max_kpt_pos_error']:.6f} m)")
            print(f"  KPT Rotation MAE: {avg_kpt_rot:.6f} rad (Max: {me['max_kpt_rot_error']:.6f} rad)")
            print(f"  Joint Position MAE: {avg_joint_pos:.6f} rad (Max: {me['max_joint_pos_error']:.6f} rad)")
            print(f"  Joint Velocity MAE: {avg_joint_vel:.6f} rad/s (Max: {me['max_joint_vel_error']:.6f} rad/s)")
            print(f"  Root Pos Error: {avg_root_pos:.3f} mm (Max: {me['max_root_pos_error']:.3f} mm)")
            print(f"  Root Vel Error: {avg_root_vel:.3f} mm/s (Max: {me['max_root_vel_error']:.3f} mm/s)")
            print(f"  Root Yaw Error: {avg_root_yaw:.6f} rad (Max: {me['max_root_yaw_error']:.6f} rad)")
            print("=" * 80 + "\n")

            if save_video and video_frames:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = VIDEO_OUTPUT_DIR / f"{file_name.replace('.npz', '')}_{stamp}.mp4"
                _save_video(video_frames, video_path, fps=int(args.freq))

            yield (
                r4((steps_done / total_steps) * 100),
                latest_length_ratio,
                avg_kpt_pos, avg_kpt_rot,
                avg_joint_pos, avg_joint_vel,
                avg_root_pos, avg_root_vel, avg_root_yaw,
                r4(me["max_kpt_pos_error"]), r4(me["max_kpt_rot_error"]),
                r4(me["max_joint_pos_error"]), r4(me["max_joint_vel_error"]),
                r4(me["max_root_pos_error"]), r4(me["max_root_vel_error"]),
                r4(me["max_root_yaw_error"]),
            )
            if run_id != _CURRENT_RUN_ID:
                return

    except Exception:
        traceback.print_exc()
        yield DEFAULT_OUTPUT


def build_interface(defaults: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="Humanoid-GPT Transformer Inference") as demo:
        gr.Markdown("# Humanoid-GPT Transformer — Gradio Inference\nLive metrics and viewer.")
        run_id_state = gr.State("")
        with gr.Row():
            onnx_track_box = gr.Textbox(label="onnx_track (ONNX)", value=defaults.onnx_track)
        with gr.Row():
            privileged_box = gr.Checkbox(label="privileged", value=defaults.privileged)
            track_dir_box = gr.Textbox(label="track_dir", value=defaults.track_dir)
            freq_box = gr.Number(label="freq (Hz)", value=defaults.freq, precision=0)
            history_len_box = gr.Number(
                label="history_len",
                value=defaults.history_len,
                precision=0,
                minimum=1,
            )

        with gr.Row():
            friction_slider = gr.Slider(
                label="Floor Friction", minimum=0.0, maximum=2.0,
                value=1.0, step=0.05,
                info="Ground friction (0.0=ice, 1.0=normal, 2.0=high)",
            )
            save_video_cb = gr.Checkbox(label="video", value=False, info="Save video per trajectory")
            show_ref_ghost_cb = gr.Checkbox(
                label="ref ghost", value=True,
                info="Overlay reference motion as a red ghost",
            )

        run_btn = gr.Button("Start Inference")
        progress_bar = gr.Slider(label="Progress (%)", minimum=0, maximum=100, value=0, step=0.0001, interactive=False)
        traj_ratio = gr.Number(label="traj_length_ratio", precision=4)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Mean Errors")
                kpt_pos = gr.Number(label="kpt_pos_error", precision=4)
                kpt_rot = gr.Number(label="kpt_rot_error", precision=4)
                joint_pos = gr.Number(label="joint_pos_error", precision=4)
                joint_vel = gr.Number(label="joint_vel_error", precision=4)
                root_pos = gr.Number(label="root_pos_error", precision=4)
                root_vel = gr.Number(label="root_vel_error", precision=4)
                root_yaw = gr.Number(label="root_yaw_error", precision=4)

            with gr.Column():
                gr.Markdown("### Max Errors")
                mkpt_pos = gr.Number(label="max_kpt_pos_error", precision=4)
                mkpt_rot = gr.Number(label="max_kpt_rot_error", precision=4)
                mjoint_pos = gr.Number(label="max_joint_pos_error", precision=4)
                mjoint_vel = gr.Number(label="max_joint_vel_error", precision=4)
                mroot_pos = gr.Number(label="max_root_pos_error", precision=4)
                mroot_vel = gr.Number(label="max_root_vel_error", precision=4)
                mroot_yaw = gr.Number(label="max_root_yaw_error", precision=4)

        outputs = [
            progress_bar, traj_ratio,
            kpt_pos, kpt_rot, joint_pos, joint_vel,
            root_pos, root_vel, root_yaw,
            mkpt_pos, mkpt_rot, mjoint_pos, mjoint_vel,
            mroot_pos, mroot_vel, mroot_yaw,
        ]
        run_btn.click(prepare_run_id, outputs=run_id_state, queue=False).then(
            run_inference_gradio,
            inputs=[
                run_id_state, onnx_track_box, privileged_box,
                track_dir_box, freq_box, history_len_box,
                friction_slider, save_video_cb, show_ref_ghost_cb,
            ],
            outputs=outputs,
            queue=True,
        )
    return demo


def _enable_queue(app: gr.Blocks, concurrency: int = 1, max_size: int | None = None):
    import inspect
    kwargs = {}
    sig = inspect.signature(app.queue)
    if "concurrency_count" in sig.parameters:
        kwargs["concurrency_count"] = concurrency
    if "max_size" in sig.parameters and max_size is not None:
        kwargs["max_size"] = max_size
    app.queue(**kwargs)


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nCtrl+C received, forcing exit...")
        os._exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    defaults = parse_cli_defaults()
    _boot_viewer_at_start(defaults)
    app = build_interface(defaults)
    _enable_queue(app, concurrency=1)
    app.launch()
