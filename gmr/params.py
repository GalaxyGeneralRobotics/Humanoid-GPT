import os
import pathlib

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parent  # Humanoid-GPT/
IK_CONFIG_ROOT = HERE / "ik_configs"

# Shared asset root.  `g1_mocap_track.xml` is the IK model: it adds the massless
# `left/right_toe_link` IK-target markers and a 0.002 s integration timestep on
# top of the revision's `g1_mjx_track.xml`, and narrows the joint ranges to the
# original GMR-galbot limits.  Those limits are a strict subset of the
# training/deploy ranges in `g1_mjx.xml` / `g1_mjx_track.xml` (verified joint by
# joint), so a retarget that respects the IK model is always reachable by the
# trained policy while the policy itself keeps the wider range to work in.
# Override the asset root with GMR_ASSET_ROOT and the robot revision with
# G1_VERSION (matches tracking.constants.G1_VERSION).
ASSET_ROOT = pathlib.Path(
    os.environ.get("GMR_ASSET_ROOT", REPO_ROOT / "storage" / "assets")
)
G1_VERSION = os.environ.get("G1_VERSION", "5010")

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / f"unitree_g1_{G1_VERSION}" / "g1_mocap_track.xml",
}

# Source-of-truth IK configs (BVH uses the original GMR-galbot weights).
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
