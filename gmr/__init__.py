"""GMR-qzk: a clean, minimal General Motion Retargeting package.

Supports retargeting human motion to the Unitree G1 from three sources:
    - "bvh"        : offline LAFAN1 / PNS BVH files
    - "fbx_noitom" : Noitom PNLink real-time mocap
    - "fbx_xsens"  : Xsens MVN real-time mocap

The retargeting engine and IK configs are extracted verbatim from the
original GMR-galbot (bvh) and Humanoid-GPT vendored (noitom/xsens) copies,
so results are bit-for-bit consistent with the originals.
"""

from .params import (
    IK_CONFIG_ROOT,
    ASSET_ROOT,
    ROBOT_XML_DICT,
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from .motion_retarget import GeneralMotionRetargeting

__all__ = [
    "GeneralMotionRetargeting",
    "IK_CONFIG_ROOT",
    "ASSET_ROOT",
    "ROBOT_XML_DICT",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
]


def __getattr__(name):
    # Lazily expose the viewer so the core package does not hard-depend on
    # mujoco.viewer / imageio when only headless retargeting is needed.
    if name == "RobotMotionViewer":
        from .robot_motion_viewer import RobotMotionViewer
        return RobotMotionViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
