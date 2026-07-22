"""Benchmark the FULL inference pipeline using fake (synthetic) sensor data.

Runs without a real robot or DDS connection, so this can be used on a
workstation as well as on the Jetson.  Useful for:
- Comparing latency across different python / numpy / scipy / onnxruntime
  versions (e.g. python 3.11 vs 3.12, numpy 1.x vs 2.x).
- Identifying per-segment overhead inside ``infer_onnx_real``.
- Sanity-checking a new ONNX model before flashing.

What this measures (per locomotion step, mode-2 / offline tracking path):

    01_tree_index_curr   ref slice for current frame
    02_tree_index_next   ref slice for next frame
    03_alloc_qpos_qvel   np.zeros + assignments inside infer_onnx_real
    04_quat2mat          scipy Rotation for gravity vector
    05_update_coord_cmd  yaw_d / xy_d encoding
    06_get_nn_state      np.hstack of all observation components
    07_onnx_run          the ONNX inference call itself
    08_nn2motor_action   conversion from action delta to motor targets
    09_infer_total       whole infer_onnx_real call (== 03..08 + bookkeeping)
    99_step_total        whole step (== 01..09)

What this does NOT measure (use ``bench_online.py`` instead):
    - DDS sensor read / low_state callback overhead
    - low_cmd packing (motor_cmd[mid].q/qd/kp/kd/tau writes)
    - CRC computation
    - DDS publish

Usage:
    python -m deploy.onboard_deploy.bench_offline
    python -m deploy.onboard_deploy.bench_offline --use-trt
    python -m deploy.onboard_deploy.bench_offline --num-iters 1000
"""

from __future__ import annotations

from deploy.onboard_deploy._bench_utils import aarch64_preload

aarch64_preload()

import time
from dataclasses import dataclass

import mujoco
import numpy as np
import tyro

from deploy.onboard_deploy._bench_utils import (
    BenchProfiler,
    print_budget_summary,
    print_environment_info,
)
from tracking.policy import Args as PolicyArgs
from tracking.policy import get_policy_onnx
from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from utils.transforms_np import quat2mat


@dataclass
class BenchArgs:
    """Full-pipeline benchmark with fake sensors (no DDS)."""

    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    policy_type: str = "mlp"
    use_trt: bool = False
    num_warmup: int = 50
    num_iters: int = 500
    freq: int = 50
    convert_xml_path: str = str(consts.TRACK_XML)
    device: str = "cuda:0"


def _tree_index(tree: dict, idx: int) -> dict:
    """Same helper as in play_track_onboard.py."""
    return {
        k: (v[idx][None] if isinstance(v, np.ndarray) else v)
        for k, v in tree.items()
    }


def make_synthetic_ref_traj(
    infer_fn: G1TrackInferFn, T: int = 200
) -> dict[str, np.ndarray]:
    """Build a (T, ...) reference trajectory with all keys infer_fn expects.

    Mirrors the output shape of ``load_offline_motions`` --> ``qpos2kpt``
    closely enough to drive ``_tree_index`` and ``infer_onnx_real`` without
    any motion data on disk.
    """
    nq = infer_fn.mj_model.nq
    nv = infer_fn.mj_model.nv
    nk = infer_fn.num_kpt

    qpos = np.tile(
        np.asarray(consts.DEFAULT_QPOS, dtype=np.float32)[None], (T, 1)
    )
    qvel = np.zeros((T, nv), dtype=np.float32)
    kpt2gv_pose = np.tile(
        np.eye(4, dtype=np.float32)[None, None], (T, nk, 1, 1)
    )
    kpt2gv_pose[..., 2, 3] = 0.78  # set z so reference height is non-zero
    kpt_cvel_in_gv = np.zeros((T, nk, 6), dtype=np.float32)
    return {
        "qpos": qpos,
        "qvel": qvel,
        "kpt2gv_pose": kpt2gv_pose,
        "kpt_cvel_in_gv": kpt_cvel_in_gv,
    }


def make_fake_sensor_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identity quaternion, zero gyro, default joint qpos, zero joint qvel."""
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_gyro = np.zeros(3, dtype=np.float32)
    jnt_qpos = np.asarray(consts.DEFAULT_QPOS[7:], dtype=np.float32).copy()
    jnt_qvel = np.zeros(consts.NUM_JOINT, dtype=np.float32)
    return root_quat, root_gyro, jnt_qpos, jnt_qvel


def step_with_profiling(
    prof: BenchProfiler,
    infer_fn: G1TrackInferFn,
    ref_traj: dict,
    track_step: int,
    sensor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """One locomotion step with per-segment profiling.

    Mirrors the mode-2 (offline tracking) path inside
    ``play_track_onboard.locomotion_step``:

        ref_curr = _tree_index(ref_traj, track_step)
        ref_next = _tree_index(ref_traj, track_step+1)
        motor_targets = infer_fn.infer_onnx_real(
            root_quat, root_gyro, jnt_qpos, jnt_qvel,
            {"ref_curr": ref_curr, "ref_next": ref_next},
        )
    """
    root_quat, root_gyro, jnt_qpos, jnt_qvel = sensor
    traj_len = len(ref_traj["qpos"])

    with prof.time("01_tree_index_curr"):
        ref_curr = _tree_index(ref_traj, track_step)
    with prof.time("02_tree_index_next"):
        nxt = min(track_step + 1, traj_len - 1)
        ref_next = _tree_index(ref_traj, nxt)

    with prof.time("09_infer_total"):
        ref_state = {"ref_curr": ref_curr, "ref_next": ref_next}
        ref_next_inner = ref_state["ref_next"]

        with prof.time("03_alloc_qpos_qvel"):
            qpos = np.zeros((1, infer_fn.mj_model.nq), dtype=np.float32)
            qpos[0, :3] = [0.0, 0.0, 0.78]
            qpos[0, 3:7] = root_quat
            qpos[0, 7:] = jnt_qpos
            qvel = np.zeros((1, infer_fn.mj_model.nv), dtype=np.float32)
            qvel[0, 3:6] = root_gyro
            qvel[0, 6:] = jnt_qvel

        with prof.time("04_quat2mat"):
            pelvis2world_rot = quat2mat(root_quat[None])
            gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]

        infer_fn.info["gyro_pelvis"][:] = root_gyro[None]
        infer_fn.info["gvec_pelvis"][:] = gvec_pelvis
        infer_fn.info["qpos"][:] = qpos
        infer_fn.info["qvel"][:] = qvel

        with prof.time("05_update_coord_cmd"):
            infer_fn.update_coord_cmd(ref_state)

        with prof.time("06_get_nn_state"):
            obs = infer_fn.get_nn_state(
                infer_fn.info, ref_next_inner, infer_fn.info["last_action"]
            )

        with prof.time("07_onnx_run"):
            nn_action = infer_fn.nn_policy.infer(obs)

        with prof.time("08_nn2motor_action"):
            motor_targets = infer_fn.nn2motor_action(nn_action)

        infer_fn.info["motor_targets"] = motor_targets.copy()
        infer_fn.info["step"] += 1
        infer_fn.info["nn_action"] = nn_action
        infer_fn.info["last_action"] = nn_action.copy()


def main(args: BenchArgs) -> None:
    print_environment_info()

    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    print(f"\nLoading policy: {args.onnx_track}")
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track,
        policy_type=args.policy_type,
        device=args.device,
    )
    track_policy = get_policy_onnx(
        policy_args, use_trt=args.use_trt, strict_trt=False
    )

    print(f"Loading MuJoCo model: {args.convert_xml_path}")
    phantom_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    phantom_model.opt.timestep = 0.001

    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)

    ref_traj = make_synthetic_ref_traj(infer_fn, T=200)
    traj_len = len(ref_traj["qpos"])
    sensor = make_fake_sensor_state()

    # ---- Warmup ----------------------------------------------------------
    print(f"\nWarmup ({args.num_warmup} iters)...")
    warmup_prof = BenchProfiler()
    for i in range(args.num_warmup):
        step_with_profiling(warmup_prof, infer_fn, ref_traj, i % traj_len, sensor)

    # ---- Benchmark -------------------------------------------------------
    prof = BenchProfiler()
    print(f"Benchmarking ({args.num_iters} iters)...")
    for i in range(args.num_iters):
        t0 = time.perf_counter()
        step_with_profiling(prof, infer_fn, ref_traj, i % traj_len, sensor)
        prof.record("99_step_total", (time.perf_counter() - t0) * 1e3)

    header = (
        f"OFFLINE pipeline latency  |  model={args.onnx_track}  "
        f"type={args.policy_type}  TRT={args.use_trt}  freq={args.freq}Hz"
    )
    print(prof.summary(header, iters=args.num_iters))

    print_budget_summary(
        np.asarray(prof.timings["99_step_total"]), target_freq_hz=args.freq
    )


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
