"""Benchmark the FULL inference pipeline on a real robot via DDS.

Reads real proprioception (joint qpos/qvel + IMU) from G1, builds the full
observation, runs ONNX inference, packs the low_cmd and computes CRC.

By default (``--no-publish-damping``) the script does NOT publish the
motor commands to DDS, so the robot does NOT move.  In that mode the
DDS publish latency is NOT measured.

To also measure the DDS publish path safely, pass ``--publish-damping``:
the script will publish damping commands (kp=0, kd=8, q=0) every step
instead of the predicted motor targets.  The robot will go limp but no
motion command from the policy ever reaches the motors.

What this measures (per locomotion step, mode-2 / offline tracking path):

    00_get_sensor_state  DDS attribute read loop (29 motors + IMU)
    01_tree_index_curr   ref slice for current frame
    02_tree_index_next   ref slice for next frame
    03_alloc_qpos_qvel   np.zeros + assignments inside infer_onnx_real
    04_quat2mat          scipy Rotation for gravity vector
    05_update_coord_cmd  yaw_d / xy_d encoding
    06_get_nn_state      np.hstack of all observation components
    07_onnx_run          the ONNX inference call itself
    08_nn2motor_action   conversion from action delta to motor targets
    09_infer_total       whole infer_onnx_real call (== 03..08 + bookkeeping)
    10_low_cmd_pack      29 attribute writes into low_cmd.motor_cmd[*]
    11_crc               CRC computation (Pack + struct + ctypes)
    12_dds_publish       (only if --publish-damping) actual DDS Write
    99_step_total        whole step (== 00..12)

Usage:
    # default: safe, no DDS write -- DDS publish latency NOT measured
    python -m deploy.onboard_deploy.bench_online --net eth0

    # also measure DDS publish (robot goes limp via damping cmds)
    python -m deploy.onboard_deploy.bench_online --net eth0 --publish-damping

NOTE: This script must be run on the on-board computer with the robot
powered up and DDS reachable on the given network interface.  It will
hang on startup if low_state never arrives.
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
from deploy.onboard_deploy.bench_offline import (
    _tree_index,
    make_synthetic_ref_traj,
)
from tracking.policy import Args as PolicyArgs
from tracking.policy import get_policy_onnx
from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
from utils.transforms_np import quat2mat


@dataclass
class BenchArgs:
    """Full-pipeline benchmark on a connected G1 (real DDS, real sensors)."""

    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    policy_type: str = "mlp"
    use_trt: bool = True
    num_warmup: int = 50
    num_iters: int = 500
    freq: int = 50
    net: str = "eth0"
    convert_xml_path: str = str(consts.TRACK_XML)
    publish_damping: bool = False
    """If True, publish damping commands (kp=0, kd=8, q=0) to DDS every
    step (robot goes limp, but DDS publish latency IS measured).
    If False, runs LowLevelControlG1 in debug=True so nothing is written
    to DDS and step 12_dds_publish is skipped."""
    fast_path: bool = True
    """Use ``infer_onnx_real_fast`` (pre-allocated buffers + IOBinding) instead
    of the inlined slow path. Required to reproduce the production deploy
    latency.  Set to False to compare against the legacy path."""


def _step_online(
    prof: BenchProfiler,
    low_ctrl,
    infer_fn: G1TrackInferFn,
    ref_traj: dict,
    track_step: int,
    publish_damping: bool,
    fast_path: bool = True,
) -> None:
    step_start = time.perf_counter()

    # ---- 00 Sensor read (DDS) -------------------------------------------
    with prof.time("00_get_sensor_state"):
        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()

    # ---- 01/02 Reference slicing ----------------------------------------
    traj_len = len(ref_traj["qpos"])
    with prof.time("01_tree_index_curr"):
        ref_curr = _tree_index(ref_traj, track_step)
    with prof.time("02_tree_index_next"):
        nxt = min(track_step + 1, traj_len - 1)
        ref_next = _tree_index(ref_traj, nxt)

    # ---- 03..09 Inline infer_onnx_real ----------------------------------
    if fast_path:
        with prof.time("09_infer_total"):
            ref_state = {"ref_curr": ref_curr, "ref_next": ref_next}
            motor_targets = infer_fn.infer_onnx_real_fast(
                root_quat, root_gyro, jnt_qpos, jnt_qvel, ref_state
            )
    else:
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

    flat_targets = np.asarray(motor_targets).flatten()

    # ---- 10 low_cmd packing ---------------------------------------------
    # Fast path: write into the byte-mirror + IDL struct simultaneously.
    # Replaces the 0.85 ms unitree SDK Pack+Trans+CRC sequence with a
    # numpy bulk write + a direct ctypes call to the C crc32_core.
    with prof.time("10_low_cmd_pack"):
        if publish_damping:
            low_ctrl.fast_pack_damping(kd_value=8.0)
        else:
            low_ctrl.fast_pack_motor_cmd(flat_targets)

    # ---- 11 CRC ----------------------------------------------------------
    with prof.time("11_crc"):
        low_ctrl.fast_compute_crc()

    # ---- 12 DDS publish (only if explicitly requested) ------------------
    if publish_damping:
        with prof.time("12_dds_publish"):
            low_ctrl._pub.Write(low_ctrl.low_cmd)

    prof.record("99_step_total", (time.perf_counter() - step_start) * 1e3)


def main(args: BenchArgs) -> None:
    print_environment_info()

    # ---- DDS ------------------------------------------------------------
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from deploy.real_robot import LowLevelControlG1

    ctrl_dt = 1.0 / args.freq

    print(f"\nDDS init on '{args.net}' ...")
    ChannelFactoryInitialize(0, args.net)
    print(
        f"Connecting to robot (DDS publish: "
        f"{'damping (real)' if args.publish_damping else 'disabled (debug=True)'}) ..."
    )
    low_ctrl = LowLevelControlG1(
        ctrl_dt=ctrl_dt, debug=(not args.publish_damping)
    )
    print("Robot connected.")

    # ---- Policy + model -------------------------------------------------
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    print(f"Loading policy: {args.onnx_track}")
    policy_args = PolicyArgs(
        onnx_track=args.onnx_track, policy_type=args.policy_type
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

    # ---- Warmup (paced at ctrl_dt) --------------------------------------
    print(f"\nWarmup ({args.num_warmup} iters)...")
    warmup_prof = BenchProfiler()
    next_t = time.perf_counter()
    for i in range(args.num_warmup):
        next_t += ctrl_dt
        _step_online(
            warmup_prof,
            low_ctrl,
            infer_fn,
            ref_traj,
            i % traj_len,
            args.publish_damping,
            args.fast_path,
        )
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # ---- Benchmark (paced at ctrl_dt) -----------------------------------
    prof = BenchProfiler()
    print(f"Benchmarking ({args.num_iters} iters)...")
    next_t = time.perf_counter()
    for i in range(args.num_iters):
        next_t += ctrl_dt
        _step_online(
            prof,
            low_ctrl,
            infer_fn,
            ref_traj,
            i % traj_len,
            args.publish_damping,
            args.fast_path,
        )
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    header = (
        f"ONLINE pipeline latency  |  net={args.net}  "
        f"publish_damping={args.publish_damping}  "
        f"TRT={args.use_trt}  freq={args.freq}Hz  "
        f"fast_path={args.fast_path}"
    )
    print(prof.summary(header, iters=args.num_iters))

    print_budget_summary(
        np.asarray(prof.timings["99_step_total"]), target_freq_hz=args.freq
    )

    # Leave the robot in damping when done
    try:
        low_ctrl.set_motor_damping()
    except Exception:
        pass


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
