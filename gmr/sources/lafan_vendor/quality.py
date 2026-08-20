"""Pose-invariant quality checks for BVH skeletons."""

import numpy as np

POSITION_SCALE_M_PER_BVH_UNIT = 0.01
# A broad adult range; unit mistakes land roughly two orders of magnitude away.
MIN_LEG_LENGTH_M = 0.5
MAX_LEG_LENGTH_M = 1.2
LEG_CHAINS = (
    ("LeftLeg", "LeftFoot"),
    ("RightLeg", "RightFoot"),
)


class SkeletonScaleError(ValueError):
    """Raised when a BVH skeleton is incompatible with the cm input contract."""


def validate_skeleton_scale(offsets, bones):
    """Validate BVH units from canonical leg offsets and return length in meters.

    The LAFAN/GMR reader expects BVH positions in centimeters. Static joint
    offsets are independent of whether the actor is standing, sitting, or
    lying down, unlike a world-Z bounding-box extent.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    bone_index = {bone: index for index, bone in enumerate(bones)}
    required = {bone for chain in LEG_CHAINS for bone in chain}
    missing = sorted(required.difference(bone_index))
    if offsets.ndim != 2 or offsets.shape[1:] != (3,) or len(offsets) != len(bones):
        raise SkeletonScaleError(
            f"degenerate skeleton (invalid offsets shape {offsets.shape})"
        )
    if missing:
        raise SkeletonScaleError(
            f"degenerate skeleton (missing canonical leg offsets: {', '.join(missing)})"
        )

    leg_lengths = [
        sum(float(np.linalg.norm(offsets[bone_index[bone]])) for bone in chain)
        for chain in LEG_CHAINS
    ]
    leg_length_m = float(np.median(leg_lengths) * POSITION_SCALE_M_PER_BVH_UNIT)
    if not np.isfinite(leg_length_m) or not (
        MIN_LEG_LENGTH_M <= leg_length_m <= MAX_LEG_LENGTH_M
    ):
        raise SkeletonScaleError(
            "degenerate skeleton (canonical leg length "
            f"{leg_length_m:.3f} m outside "
            f"[{MIN_LEG_LENGTH_M:.3f}, {MAX_LEG_LENGTH_M:.3f}] m; "
            "expected BVH positions in centimeters)"
        )
    return leg_length_m
