"""Offline BVH (LAFAN1 / PNS) source loader.

Verbatim port of GMR-galbot's ``utils/lafan1.py`` (the source of truth for
the bvh tracking path), with imports rewired to this package. Returns a list
of per-frame dicts ``{body_name: (position, orientation_wxyz)}`` ready to feed
into ``GeneralMotionRetargeting(src_human="bvh").retarget(frame)``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import lafan_vendor as _lv
from .lafan_vendor import utils
from .lafan_vendor.extract import read_bvh
from .lafan_vendor.quality import validate_skeleton_scale


def load_lafan1_file(bvh_file, pns=False):
    """Load a BVH file into per-frame global pose dicts.

    Args:
        bvh_file: path to the .bvh file.
        pns: set True for Noitom PNS-format BVH (space-delimited channels);
            False for LAFAN1-format BVH.

    Returns:
        (frames, human_height) where frames is a list of dicts
        {body_name: (position[3], orientation_wxyz[4])}.
    """
    data = read_bvh(bvh_file, pns=pns)
    validate_skeleton_scale(data.offsets, data.bones)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    rotation_inv_quat = R.from_matrix(rotation_matrix).inv().as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            orientation = utils.quat_mul(orientation, rotation_inv_quat)
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])

        frames.append(result)

    human_height = 1.75

    return frames, human_height
