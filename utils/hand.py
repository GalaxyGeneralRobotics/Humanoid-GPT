"""End-effector (hand) configuration + hand-aware MuJoCo model loading.

The ``HAND`` environment variable selects which end-effector is mounted on the
Unitree G1 wrists, analogous to ``G1_VERSION`` / ``BASE_KP_KD_SCALE``.  It is
honoured everywhere a robot model is loaded (training, inference, evaluation,
deployment, visualization).

    HAND=None      bare wrist, no hand at all
    HAND=Default   lightweight rubber dummy hand (the default; what ships baked
                   into unitree_g1_5010 / unitree_g1_4010)
    HAND=Dex3      Unitree Dex3-1 dexterous hand (7 DoF / hand)
    HAND=BrainCo   BrainCo (强脑) 5-finger dexterous hand

Design notes
------------
* The whole-body tracking policy always controls the 29 G1 body joints and never
  the fingers, so the model used for training/inference must keep ``nq == 36``
  (free joint + 29 joints).  Therefore the hand is attached *rigidly* by default:
  the finger joints are stripped (a jointless child body in MuJoCo is welded to
  its parent yet keeps its mass + collision and adds **zero** DoF).
* The hand assets live in ``storage/assets/hand/<variant>/`` and are version
  independent (they attach onto either G1 hardware version at the
  ``*_wrist_yaw_link`` frame).  Each ``hand.xml`` keeps the finger joints so the
  same asset can be used *articulated* (``rigid=False``) for teleoperation /
  grasping, where the fingers are driven by a separate hand controller.
* For ``HAND=Default`` the loader simply returns the standalone version model
  unchanged, so existing checkpoints / results are bit-for-bit unaffected.
"""

from __future__ import annotations

import os

import mujoco

from utils.path import PATH_ASSET

# ---------------------------------------------------------------------------
# HAND environment variable
# ---------------------------------------------------------------------------
VALID_HANDS = ("None", "Default", "Dex3", "BrainCo")

HAND = os.environ.get("HAND", "Default")
if HAND not in VALID_HANDS:
    raise ValueError(
        f"Invalid HAND={HAND!r}. Choose one of {VALID_HANDS} "
        f"(set via the HAND environment variable, default 'Default')."
    )

# variant -> asset folder name under storage/assets/hand/
_HAND_DIR = {
    "None": "none",
    "Default": "default",
    "Dex3": "dex3",
    "BrainCo": "brainco",
}

# Per-variant deployment metadata.
#   controller : key used by deploy code to pick the real-robot hand controller
#   ctrl_dof   : actuated command DoF per hand on the real robot
#   has_hand   : whether a physical hand is mounted
HAND_INFO = {
    "None":    {"controller": None,      "ctrl_dof": 0, "has_hand": False},
    "Default": {"controller": None,      "ctrl_dof": 0, "has_hand": False},
    "Dex3":    {"controller": "dex3",    "ctrl_dof": 7, "has_hand": True},
    "BrainCo": {"controller": "brainco", "ctrl_dof": 6, "has_hand": True},
}


def hand_dir(hand: str = HAND):
    """Return the asset directory for a hand variant."""
    return PATH_ASSET / "hand" / _HAND_DIR[hand]


def hand_asset_xml(hand: str = HAND):
    """Return the standalone ``hand.xml`` for a hand variant (None has none)."""
    return hand_dir(hand) / "hand.xml"


# ---------------------------------------------------------------------------
# Model surgery helpers (operate on a mujoco.MjSpec)
# ---------------------------------------------------------------------------
_WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _has_body(spec: "mujoco.MjSpec", name: str) -> bool:
    try:
        return spec.body(name) is not None
    except (KeyError, ValueError):
        return False


def _strip_baked_hand(spec: "mujoco.MjSpec") -> None:
    """Remove the rubber dummy hand (visual mesh, collision proxy, palm site)
    that ships baked into the version models, leaving a bare wrist."""
    for side in ("left", "right"):
        wrist = f"{side}_wrist_yaw_link"
        if not _has_body(spec, wrist):
            continue
        body = spec.body(wrist)
        for g in list(body.geoms):
            is_rubber = bool(g.meshname) and "rubber_hand" in g.meshname
            if is_rubber or g.name == f"{side}_hand_collision":
                spec.delete(g)
        for s in list(body.sites):
            if s.name == f"{side}_palm":
                spec.delete(s)


def _remove_hand_contacts(spec: "mujoco.MjSpec") -> None:
    """Drop explicit contact pairs that reference a hand collision geom."""
    for p in list(spec.pairs):
        g1 = p.geomname1 or ""
        g2 = p.geomname2 or ""
        if "hand_collision" in g1 or "hand_collision" in g2:
            spec.delete(p)


def _side_hand_spec(hand: str, side: str, rigid: bool) -> "mujoco.MjSpec":
    """Build a single-side hand spec (the other side's body + meshes removed so
    that left and right can be attached without mesh-id collisions)."""
    other = "right" if side == "left" else "left"
    hd = hand_dir(hand)
    hs = mujoco.MjSpec.from_file(str(hd / "hand.xml"))
    # Resolve meshes against an absolute path so they survive attachment.
    hs.meshdir = str((hd / "meshes").resolve())
    hs.delete(hs.body(f"{other}_hand"))
    for mesh in list(hs.meshes):
        if mesh.name.startswith(f"{other}_"):
            hs.delete(mesh)
    if rigid:
        for j in list(hs.joints):
            hs.delete(j)
    return hs


def _drop_keyframes(spec: "mujoco.MjSpec") -> None:
    """Articulated hands change nq, invalidating the (size-36) keyframes."""
    keys = getattr(spec, "keys", None)
    if keys is None:
        return
    for k in list(keys):
        spec.delete(k)


def apply_hand(spec: "mujoco.MjSpec", hand: str = HAND, rigid: bool = True) -> "mujoco.MjSpec":
    """Mutate ``spec`` in place so the wrists carry the requested ``hand``.

    Args:
        spec: a loaded ``mujoco.MjSpec`` (e.g. ``MjSpec.from_file(scene_xml)``).
        hand: one of :data:`VALID_HANDS`.
        rigid: if True (default), strip finger joints so the hand is a rigid
            attachment that keeps the model at 29 joints (nq unchanged).  If
            False, keep the finger joints (articulated hand, extra DoF) for
            teleoperation / grasping use cases.
    Returns:
        The same ``spec`` (for chaining).
    """
    if hand == "Default":
        return spec

    _strip_baked_hand(spec)

    if hand == "None":
        _remove_hand_contacts(spec)
        return spec

    # Dex3 / BrainCo: attach the dexterous hand onto each wrist.
    for side in ("left", "right"):
        wrist = f"{side}_wrist_yaw_link"
        if not _has_body(spec, wrist):
            continue
        hs = _side_hand_spec(hand, side, rigid=rigid)
        frame = spec.body(wrist).add_frame()
        frame.attach_body(hs.body(f"{side}_hand"), "", "")

    if not rigid:
        _drop_keyframes(spec)
    return spec


def load_mj_spec(xml_path, hand: str = HAND, rigid: bool = True) -> "mujoco.MjSpec":
    """Load ``xml_path`` into an ``MjSpec`` with the requested hand applied."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    return apply_hand(spec, hand=hand, rigid=rigid)


def load_mj_model(xml_path, hand: str = HAND, rigid: bool = True) -> "mujoco.MjModel":
    """Hand-aware drop-in replacement for ``mujoco.MjModel.from_xml_path``.

    For ``HAND=Default`` this is exactly ``mujoco.MjModel.from_xml_path`` so the
    default behaviour / existing checkpoints are bit-for-bit unchanged.
    """
    if hand == "Default":
        return mujoco.MjModel.from_xml_path(str(xml_path))
    return load_mj_spec(xml_path, hand=hand, rigid=rigid).compile()
