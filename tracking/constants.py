import os
from pathlib import Path
import numpy as np
from utils.path import PATH_ASSET
# HAND selects the mounted end-effector (None / Default / Dex3 / BrainCo) and is
# honoured by the hand-aware model loader (utils.hand.load_mj_model). Re-exported
# here so the rest of the training stack can read ``consts.HAND``.
from utils.hand import HAND, HAND_INFO, load_mj_model, load_mj_spec  # noqa: F401

G1_VERSION = os.environ.get("G1_VERSION", "5010")
ROOT_PATH = PATH_ASSET / f"unitree_g1_{G1_VERSION}"
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_rough_terrain.xml"
FULL_COLLISIONS_XML = ROOT_PATH / "scene_mjx.xml"
BODY_POSE_XML = ROOT_PATH / "scene_mjx_body_pose.xml"
LOCO_XML = ROOT_PATH / "scene_mjx_loco.xml"
TRACK_XML = ROOT_PATH / "scene_mjx_track.xml"
DEBUG_TRACK_XML = ROOT_PATH / "scene_mjx_track_debug.xml"


def task_to_xml(task_name: str) -> Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
        "full_collision": FULL_COLLISIONS_XML,
        "body_pose": BODY_POSE_XML,
        "locomotion_flat": LOCO_XML,
        "track": TRACK_XML,
    }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = [
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
]
RIGHT_FEET_GEOMS = [
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
]

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

#############
#   JOINT   #
#############

NUM_JOINT = 29


class MotorID:
    LEG_L = [0, 1, 2, 3, 4, 5]
    LEG_R = [6, 7, 8, 9, 10, 11]
    WAIST = [12, 13, 14]
    ARM_L = [15, 16, 17, 18, 19, 20, 21]
    ARM_R = [22, 23, 24, 25, 26, 27, 28]
    LEGs = LEG_L + LEG_R
    ARMs = ARM_L + ARM_R
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R


class MotorName:
    LEG_L = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    LEG_R = [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    WAIST = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    ARM_L = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    ARM_R = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R


RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.7, 0.7),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Right leg.
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.7, 0.7),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Waist.
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    # Left shoulder.
    (-1.57, 0.520),
    (0.000, 1.570),
    (-1.57, 0.000),
    (-0.52, 1.570),
    (-1.57, 1.570),
    (-0.52, 0.520),
    (-0.52, 0.520),
    # Right shoulder.
    (-1.57, 0.520),
    (-1.57, 0.000),
    (0.000, 1.570),
    (-0.52, 1.570),
    (-1.57, 1.570),
    (-0.52, 0.520),
    (-0.52, 0.520),
)


DEFAULT_CHEST_Z = 1.05

TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])

DEFAULT_QPOS = np.float32([
	0, 0, 0.78,      # base xyz
	1, 0, 0, 0,     # base quat (w, x, y, z)
	-0.1, 0, 0, 0.3, -0.2, 0,    # left leg
	-0.1, 0, 0, 0.3, -0.2, 0,    # right leg
	0, 0, 0,                      # waist (yaw, roll, pitch but only yaw used)
	0.2, 0.3, 0, 1.28, 0, 0, 0,   # left arm (only pitch, roll, yaw, elbow, wrist_roll used)
	0.2,-0.3, 0, 1.28, 0, 0, 0,   # right arm
])


BASE_KPs = np.float32([
    40.17923737, 99.09842682, 40.17923737, 99.09842682, 28.5012455 ,
    28.5012455 , 40.17923737, 99.09842682, 40.17923737, 99.09842682,
    28.5012455 , 28.5012455 , 40.17923737, 28.5012455 , 28.5012455 ,
    14.25062275, 14.25062275, 14.25062275, 14.25062275, 14.25062275,
    16.77832794, 16.77832794, 14.25062275, 14.25062275, 14.25062275,
    14.25062275, 14.25062275, 16.77832794, 16.77832794
])

BASE_KDs = np.float32([
    2.5578897 , 6.30880165, 2.5578897 , 6.30880165, 1.81444573,
    1.81444573, 2.5578897 , 6.30880165, 2.5578897 , 6.30880165,
    1.81444573, 1.81444573, 2.5578897 , 1.81444573, 1.81444573,
    0.90722287, 0.90722287, 0.90722287, 0.90722287, 0.90722287,
    1.06814146, 1.06814146, 0.90722287, 0.90722287, 0.90722287,
    0.90722287, 0.90722287, 1.06814146, 1.06814146
])

UPPER_BODY_JOINTs = [
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

KPT_LOWER = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]
KPT_UPPER = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
KPT_NAMES = [
    # lower body
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    # upper body
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
KPT_END_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

FEET_LINKs = ["left_ankle_roll_link", "right_ankle_roll_link"]

DOF_VEL_LIMITS = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0
]

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
    # -------------- tracking only --------------
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
OBS_JOINT_NAMES = ACTION_JOINT_NAMES

ACTION_JOINT_NAMES_66177 = [
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
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",  # 20
    "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",  # 27
    "right_wrist_yaw_joint",  # 28
]


ACTION_JOINT_NAMES_66155 = [
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
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # "left_wrist_pitch_joint",  # 20
    # "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    # "right_wrist_pitch_joint",  # 27
    # "right_wrist_yaw_joint",  # 28
]


ACTION_JOINT_NAMES_66144 = [
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
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",  # 13
    # "waist_pitch_joint",  # 14
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # "left_wrist_roll_joint",
    # "left_wrist_pitch_joint",  # 20
    # "left_wrist_yaw_joint",  # 21
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    # "right_wrist_roll_joint",
    # "right_wrist_pitch_joint",  # 27
    # "right_wrist_yaw_joint",  # 28
]

BASE_KP_KD_SCALE = float(os.environ.get("BASE_KP_KD_SCALE", "1.0"))

JOINT_KP_KD_SCALE = {
    # left leg
    "left_hip_pitch_joint": 1,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 1,
    "left_knee_joint": 1,
    "left_ankle_pitch_joint": 1,
    "left_ankle_roll_joint": 1,
    # right leg
    "right_hip_pitch_joint": 1,
    "right_hip_roll_joint": 1,
    "right_hip_yaw_joint": 1,
    "right_knee_joint": 1,
    "right_ankle_pitch_joint": 1,
    "right_ankle_roll_joint": 1,
    # waist
    "waist_yaw_joint": 1,
    "waist_roll_joint": 1,
    "waist_pitch_joint": 1,
    # left arm
    "left_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 1,
    "left_shoulder_yaw_joint": 1,
    "left_elbow_joint": 1,
    "left_wrist_roll_joint": 1,
    "left_wrist_pitch_joint": 1,
    "left_wrist_yaw_joint": 1,
    # right arm
    "right_shoulder_pitch_joint": 1,
    "right_shoulder_roll_joint": 1,
    "right_shoulder_yaw_joint": 1,
    "right_elbow_joint": 1,
    "right_wrist_roll_joint": 1,
    "right_wrist_pitch_joint": 1,
    "right_wrist_yaw_joint": 1,
}

JOINT_KP_KD_SCALE = np.array(list(JOINT_KP_KD_SCALE.values()), dtype=np.float32)
KPs = BASE_KPs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE ** 2
KDs = BASE_KDs * JOINT_KP_KD_SCALE * BASE_KP_KD_SCALE
ACTION_SCALE = TORQUE_LIMIT / KPs
