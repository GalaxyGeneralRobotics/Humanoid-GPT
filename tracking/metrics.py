"""Metrics calculation functions for robot tracking evaluation."""

import numpy as np
import tree
from scipy.spatial.transform import Rotation as R
from utils.transforms_np import quat2mat, batch_base2navi, quat2euler, WarpPi


class TempState:
    def __init__(self, state_data):
        self.mj_data = type('obj', (object,), {})()
        self.mj_data.qpos = state_data['qpos']
        self.mj_data.qvel = state_data['qvel']
        self.mj_data.xpos = state_data['xpos']
        self.mj_data.xmat = state_data['xmat']


def calculate_kpt_mae_error(state, ref_curr, ref_next, mj_model):
    """Calculate keypoint Mean Absolute Error for position and rotation."""
    # Get current robot state
    qpos = state.mj_data.qpos
    pelvis2world_rot = quat2mat(qpos[3:7][None])
    pelvis2world_pos = qpos[:3][None]
    
    # Get keypoint body IDs (same as in training)
    KPT_NAMES = [
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"
    ]
    body_ids_kpt = np.array([mj_model.body(n).id for n in KPT_NAMES])
    
    # Get current keypoint poses
    kpt2wrd_rot = state.mj_data.xmat[body_ids_kpt].reshape(-1, 3, 3)
    kpt2wrd_pos = state.mj_data.xpos[body_ids_kpt][None]
    
    # Transform to navigation frame
    navi2world_rot = batch_base2navi(pelvis2world_rot)
    navi2world_pose = np.eye(4)
    navi2world_pose[:2, 3] = qpos[:2]
    navi2world_pose[:3, :3] = navi2world_rot[0]
    
    kpt2wrd_pose = np.full((len(KPT_NAMES), 4, 4), np.eye(4))
    kpt2wrd_pose[:, :3, :3] = kpt2wrd_rot
    kpt2wrd_pose[:, :3, 3] = kpt2wrd_pos[0]
    kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
    navi2kpt_pose = np.linalg.inv(kpt2navi_pose)
    
    # Calculate error with reference
    curr_ref2kpt_pose = navi2kpt_pose @ ref_curr["kpt2gv_pose"][0]
    
    # Position error (MAE)
    pos_error = np.abs(curr_ref2kpt_pose[:, :3, 3])
    pos_mae = np.mean(pos_error)
    
    # Rotation error (MAE of rotation angles)
    rot_error = curr_ref2kpt_pose[:, :3, :3]
    rotvec = R.from_matrix(rot_error.reshape(-1, 3, 3)).as_rotvec().reshape(len(KPT_NAMES), 3)
    rot_angle_error = np.linalg.norm(rotvec, axis=1)
    rot_mae = np.mean(rot_angle_error)
    
    return pos_mae, rot_mae


def calculate_joint_tracking_error(state, ref_curr):
    """Calculate joint position and velocity tracking errors."""
    # Get current joint positions and velocities
    curr_qpos = state.mj_data.qpos[7:]  # Skip base position/orientation
    curr_qvel = state.mj_data.qvel[6:]  # Skip base velocity
    
    # Get reference joint positions and velocities
    ref_qpos = ref_curr["qpos"][0, 7:]
    ref_qvel = ref_curr["qvel"][0, 6:]
    
    # Calculate errors
    pos_error = np.abs(curr_qpos - ref_qpos)
    vel_error = np.abs(curr_qvel - ref_qvel)
    
    pos_mae = np.mean(pos_error)
    vel_mae = np.mean(vel_error)
    
    return pos_mae, vel_mae


def calculate_root_tracking_error(state, ref_curr):
    """Calculate root (base) position and velocity errors in mm/mm/s."""
    curr_root_pos = state.mj_data.qpos[:3]
    curr_root_vel = state.mj_data.qvel[:3]
    cur_root_yaw = quat2euler(state.mj_data.qpos[3:7])[2]

    ref_root_pos = ref_curr["qpos"][0, :3]
    ref_root_vel = ref_curr["qvel"][0, :3]
    ref_root_yaw = quat2euler(ref_curr["qpos"][0, 3:7])[2]

    pos_error_m = np.mean(np.abs(curr_root_pos - ref_root_pos)) * 1000.0
    vel_error_ms = np.mean(np.abs(curr_root_vel - ref_root_vel)) * 1000.0
    yaw_error = np.mean(np.abs(WarpPi(cur_root_yaw - ref_root_yaw)))

    return pos_error_m, vel_error_ms, yaw_error


def calculate_joint_jerk(state_history, ctrl_dt=0.02):
    """Calculate the mean joint jerk (rad/s^3) over a recorded trajectory.

    Jerk is the 3rd-order finite difference of the joint angles (``qpos[7:]``)
    sampled at the control rate ``ctrl_dt``. It complements per-frame pose
    tracking errors (KPT / joint MAE) by capturing how smooth and
    hardware-friendly the produced motion is: a controller can attain a low
    tracking error while still producing very jittery, high-jerk motion that is
    unsafe to deploy on real hardware.

    Args:
        state_history: list of per-step dicts, each containing ``qpos`` (the
            full MuJoCo joint position vector, base pose in the first 7 dims).
        ctrl_dt: control timestep in seconds (e.g. 0.02 for a 50 Hz policy).

    Returns:
        Mean absolute joint jerk in rad/s^3. Returns ``inf`` when the history is
        too short (< 4 frames) or ``ctrl_dt`` is non-positive.
    """
    if not state_history or ctrl_dt <= 0.0:
        return float("inf")
    try:
        qpos = np.stack(
            [np.asarray(s["qpos"][7:], dtype=np.float64) for s in state_history]
        )
    except Exception:
        return float("inf")
    if qpos.shape[0] < 4:
        return float("inf")

    # 2nd-order central finite difference -> joint acceleration (rad/s^2):
    #   a[t] = (q[t+1] - 2 q[t] + q[t-1]) / dt^2
    accel = (qpos[2:] - 2.0 * qpos[1:-1] + qpos[:-2]) / (ctrl_dt ** 2)
    # forward difference of acceleration -> joint jerk (rad/s^3):
    #   j[t] = (a[t+1] - a[t]) / dt
    jerk = np.diff(accel, axis=0) / ctrl_dt
    return float(np.mean(np.abs(jerk)))


def calculate_max_errors(traj_metrics):
    """Calculate maximum errors across all frames from trajectory metrics.
    
    Args:
        traj_metrics: Dictionary containing error lists:
            - kpt_pos_errors: List of keypoint position errors per frame
            - kpt_rot_errors: List of keypoint rotation errors per frame
            - joint_pos_errors: List of joint position errors per frame
            - joint_vel_errors: List of joint velocity errors per frame
            - root_pos_errors: List of root position errors per frame
            - root_vel_errors: List of root velocity errors per frame
            - root_yaw_errors: List of root yaw errors per frame
    
    Returns:
        Dictionary containing max errors:
            - max_kpt_pos_error: Maximum keypoint position error
            - max_kpt_rot_error: Maximum keypoint rotation error
            - max_joint_pos_error: Maximum joint position error
            - max_joint_vel_error: Maximum joint velocity error
            - max_root_pos_error: Maximum root position error
            - max_root_vel_error: Maximum root velocity error
            - max_root_yaw_error: Maximum root yaw error
    """
    max_errors = {}
    
    if traj_metrics.get("kpt_pos_errors") and len(traj_metrics["kpt_pos_errors"]) > 0:
        max_errors["max_kpt_pos_error"] = np.max(traj_metrics["kpt_pos_errors"])
    else:
        max_errors["max_kpt_pos_error"] = 0.0
    
    if traj_metrics.get("kpt_rot_errors") and len(traj_metrics["kpt_rot_errors"]) > 0:
        max_errors["max_kpt_rot_error"] = np.max(traj_metrics["kpt_rot_errors"])
    else:
        max_errors["max_kpt_rot_error"] = 0.0
    
    if traj_metrics.get("joint_pos_errors") and len(traj_metrics["joint_pos_errors"]) > 0:
        max_errors["max_joint_pos_error"] = np.max(traj_metrics["joint_pos_errors"])
    else:
        max_errors["max_joint_pos_error"] = 0.0
    
    if traj_metrics.get("joint_vel_errors") and len(traj_metrics["joint_vel_errors"]) > 0:
        max_errors["max_joint_vel_error"] = np.max(traj_metrics["joint_vel_errors"])
    else:
        max_errors["max_joint_vel_error"] = 0.0
    
    if traj_metrics.get("root_pos_errors") and len(traj_metrics["root_pos_errors"]) > 0:
        max_errors["max_root_pos_error"] = np.max(traj_metrics["root_pos_errors"])
    else:
        max_errors["max_root_pos_error"] = 0.0
    
    if traj_metrics.get("root_vel_errors") and len(traj_metrics["root_vel_errors"]) > 0:
        max_errors["max_root_vel_error"] = np.max(traj_metrics["root_vel_errors"])
    else:
        max_errors["max_root_vel_error"] = 0.0
    
    if traj_metrics.get("root_yaw_errors") and len(traj_metrics["root_yaw_errors"]) > 0:
        max_errors["max_root_yaw_error"] = np.max(traj_metrics["root_yaw_errors"])
    else:
        max_errors["max_root_yaw_error"] = 0.0
    
    return max_errors


def check_termination_conditions(state, ref_curr, mj_model, alignment_transform=None):
    """Check termination conditions using G1TrackInferFn's logic with custom thresholds."""
    # Get current robot state
    qpos = state.mj_data.qpos
    pelvis2world_rot = quat2mat(qpos[3:7][None])
    pelvis2world_pos = qpos[:3][None]

    # Get keypoint body IDs (same as G1TrackInferFn)
    KPT_NAMES = [
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"
    ]
    body_ids_kpt = np.array([mj_model.body(n).id for n in KPT_NAMES])

    # Get current keypoint poses
    kpt2wrd_rot = state.mj_data.xmat[body_ids_kpt].reshape(-1, 3, 3)
    kpt2wrd_pos = state.mj_data.xpos[body_ids_kpt][None]

    # Transform to navigation frame (same as G1TrackInferFn)
    navi2world_rot = batch_base2navi(pelvis2world_rot)
    navi2world_pose = np.full((1, 4, 4), np.eye(4))
    navi2world_pose[0, :3, :3] = navi2world_rot[0]
    navi2world_pose[0, :2, 3] = pelvis2world_pos[0, :2]

    kpt2wrd_pose = np.full((1, len(KPT_NAMES), 4, 4), np.eye(4))
    kpt2wrd_pose[0, :, :3, :3] = kpt2wrd_rot
    kpt2wrd_pose[0, :, :3, 3] = kpt2wrd_pos[0]
    kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
    navi2kpt_pose = np.linalg.inv(kpt2navi_pose)

    # Calculate error with reference (same as G1TrackInferFn)
    curr_ref2kpt_pose = navi2kpt_pose @ ref_curr["kpt2gv_pose"]

    # Apply alignment transform if provided
    if alignment_transform is not None:
        curr_ref2kpt_pose = curr_ref2kpt_pose @ np.linalg.inv(alignment_transform)

    # Termination conditions with custom thresholds
    # Pelvis/torso rotation error (same logic as G1TrackInferFn but different threshold)
    kpt_id_peltor = [0, 7]  # pelvis and torso indices
    err_rot = curr_ref2kpt_pose[0, kpt_id_peltor, :3, :3]
    rotvec = R.from_matrix(err_rot.reshape(-1, 3, 3)).as_rotvec().reshape(len(kpt_id_peltor), 3)
    theta = np.linalg.norm(rotvec, axis=1)
    done_rot = (theta > 1).any()

    # Height error check: use z-axis error instead of absolute z position
    # This avoids false positives for squatting or ground-level actions
    err_height = np.abs(curr_ref2kpt_pose[0, kpt_id_peltor, 2, 3])
    done_height = (err_height > 0.25).any()  # m

    # NaN check
    done_nan = np.isnan(qpos).any() or np.isnan(state.mj_data.qvel).any()

    # Determine termination reason
    termination_reason = None
    if done_height:
        max_err_height = np.max(err_height)
        termination_reason = f"Height error (max z error: {max_err_height:.3f}m > 0.25m)"
    elif done_rot:
        max_theta = np.max(theta)
        termination_reason = f"Rotation error (max angle: {max_theta:.3f}rad > 1.0rad)"
    elif done_nan:
        termination_reason = "NaN values detected in joint positions/velocities"

    is_terminated = done_height or done_rot or done_nan
    
    return is_terminated, termination_reason


def calculate_trajectory_length(state_history, ref_traj, mj_model):
    """Calculate trajectory length as a ratio (0-1) based on termination conditions."""
    if len(state_history) == 0:
        return 0.0, 0

    # Get the actual trajectory length from reference data
    actual_trajectory_length = len(ref_traj["qpos"])

    # Check when termination occurs
    termination_step = len(state_history)  # Default to full length if no early termination

    # Calculate coordinate system alignment transform from first step
    if len(state_history) > 0:
        first_state_data = state_history[0]
        first_ref = tree.map_structure(lambda x: x[0][None], ref_traj)

        # Get current robot state
        qpos = first_state_data['qpos']
        pelvis2world_rot = quat2mat(qpos[3:7][None])

        # Get keypoint body IDs
        KPT_NAMES = [
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"
        ]
        body_ids_kpt = np.array([mj_model.body(n).id for n in KPT_NAMES])

        # Get current keypoint poses
        kpt2wrd_rot = first_state_data['xmat'][body_ids_kpt].reshape(-1, 3, 3)
        kpt2wrd_pos = first_state_data['xpos'][body_ids_kpt][None]

        # Transform to navigation frame
        navi2world_rot = batch_base2navi(pelvis2world_rot)
        navi2world_pose = np.eye(4)
        navi2world_pose[:2, 3] = qpos[:2]
        navi2world_pose[:3, :3] = navi2world_rot[0]

        kpt2wrd_pose = np.full((len(KPT_NAMES), 4, 4), np.eye(4))
        kpt2wrd_pose[:, :3, :3] = kpt2wrd_rot
        kpt2wrd_pose[:, :3, 3] = kpt2wrd_pos[0]
        kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
        navi2kpt_pose = np.linalg.inv(kpt2navi_pose)

        # Calculate alignment transform: ref_navi2curr_navi
        curr_ref2kpt_pose = navi2kpt_pose @ first_ref["kpt2gv_pose"][0]
        alignment_transform = curr_ref2kpt_pose  # This is the transform from ref to current

    for i, state_data in enumerate(state_history):
        ref_curr = tree.map_structure(lambda x: x[i][None], ref_traj)

        temp_state = TempState(state_data)
        is_terminated, _ = check_termination_conditions(temp_state, ref_curr, mj_model, alignment_transform=alignment_transform)
        if is_terminated:
            termination_step = i + 1
            break

    # Calculate length as ratio (0-1) based on actual trajectory length
    trajectory_length_ratio = termination_step / actual_trajectory_length

    return min(trajectory_length_ratio, 1.0), termination_step
