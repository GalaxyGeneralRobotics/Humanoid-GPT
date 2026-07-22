import os
import pathlib

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parent  # Humanoid-GPT/
IK_CONFIG_ROOT = HERE / "ik_configs"

# Shared asset root. Defaults to Humanoid-GPT/storage/assets so GMR retargets
# onto the *same* robot model Humanoid-GPT trains/deploys on. Override the asset
# root with GMR_ASSET_ROOT and the robot revision with G1_VERSION (matches
# tracking.constants.G1_VERSION).
ASSET_ROOT = pathlib.Path(
    os.environ.get("GMR_ASSET_ROOT", REPO_ROOT / "storage" / "assets")
)
G1_VERSION = os.environ.get("G1_VERSION", "5010")

# GMR retargets directly onto the Humanoid-GPT training/deploy kinematics.
# `g1_mocap_track.xml` is generated from that version's `g1_mjx_track.xml`
# (identical joints/axes/orientations) by adding the massless `left/right_
# toe_link` IK-target markers and setting the IK integration timestep to
# 0.002. It reuses the same meshes, so there is no separate robot model.
ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / f"unitree_g1_{G1_VERSION}" / "g1_mocap_track.xml",
}

# Source-of-truth IK configs (bvh <- GMR-galbot; fbx_* <- Humanoid-GPT vendored)
IK_CONFIG_DICT = {
    "bvh": {
        "unitree_g1": IK_CONFIG_ROOT / "bvh_to_g1.json",
    },
    "fbx_noitom": {
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1_noitom.json",
    },
    "fbx_xsens": {
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1_xsens.json",
    },
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
}
