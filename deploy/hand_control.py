"""Dex3-1 hand controller for Unitree G1.

Provides open/close control for left and right dexterous hands via the
Unitree SDK DDS interface.  All hardware imports are deferred so that
simulation-only usage never requires the SDK.
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum

from deploy.constants import HAND, HAND_INFO

# Predefined hand poses
HAND_POSES = {
    "left": {
        "open":  np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "close": np.array([0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74], dtype=np.float32),
    },
    "right": {
        "open":  np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "close": np.array([0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74], dtype=np.float32),
    },
}

NUM_HAND_MOTORS = 7

TOPIC_LEFT_CMD = "rt/dex3/left/cmd"
TOPIC_RIGHT_CMD = "rt/dex3/right/cmd"
TOPIC_LEFT_STATE = "rt/dex3/left/state"
TOPIC_RIGHT_STATE = "rt/dex3/right/state"


class _LeftJoint(IntEnum):
    Thumb0 = 0; Thumb1 = 1; Thumb2 = 2
    Middle0 = 3; Middle1 = 4
    Index0 = 5; Index1 = 6


class _RightJoint(IntEnum):
    Thumb0 = 0; Thumb1 = 1; Thumb2 = 2
    Index0 = 3; Index1 = 4
    Middle0 = 5; Middle1 = 6


class Dex3Controller:
    """Unitree Dex3-1 dual-hand controller."""

    def __init__(self, net: str, re_init: bool = True):
        from unitree_sdk2py.core.channel import (
            ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

        if re_init:
            ChannelFactoryInitialize(0, net)

        self._left_pub = ChannelPublisher(TOPIC_LEFT_CMD, HandCmd_)
        self._left_pub.Init()
        self._right_pub = ChannelPublisher(TOPIC_RIGHT_CMD, HandCmd_)
        self._right_pub.Init()

        self._left_sub = ChannelSubscriber(TOPIC_LEFT_STATE, HandState_)
        self._left_sub.Init()
        self._right_sub = ChannelSubscriber(TOPIC_RIGHT_STATE, HandState_)
        self._right_sub.Init()

        self._left_msg = unitree_hg_msg_dds__HandCmd_()
        self._right_msg = unitree_hg_msg_dds__HandCmd_()
        self._init_msgs()
        print("[Dex3Controller] Initialized.")

    def _init_msgs(self):
        kp, kd = 0.3, 0.1
        for joint_enum, msg, defaults in [
            (_LeftJoint, self._left_msg, np.zeros(NUM_HAND_MOTORS)),
            (_RightJoint, self._right_msg, np.zeros(NUM_HAND_MOTORS)),
        ]:
            for jid in joint_enum:
                mode = ((jid & 0x0F) | (0x03 << 4)) & 0xFF
                msg.motor_cmd[jid].mode = mode
                msg.motor_cmd[jid].q = defaults[jid]
                msg.motor_cmd[jid].dq = 0.0
                msg.motor_cmd[jid].tau = 0.0
                msg.motor_cmd[jid].kp = kp
                msg.motor_cmd[jid].kd = kd
        self._left_pub.Write(self._left_msg)
        self._right_pub.Write(self._right_msg)

    def ctrl_dual_hand(self, left_q: np.ndarray, right_q: np.ndarray):
        for jid in _LeftJoint:
            self._left_msg.motor_cmd[jid].q = left_q[jid]
        for jid in _RightJoint:
            self._right_msg.motor_cmd[jid].q = right_q[jid]
        self._left_pub.Write(self._left_msg)
        self._right_pub.Write(self._right_msg)


def update_hand_from_mocap(
    hand_ctrl: "Dex3Controller | None",
    hand_cmd: tuple | None,
    last_left: bool | None,
    last_right: bool | None,
) -> tuple[bool | None, bool | None]:
    """Send open/close commands only on state change (debounce).

    Args:
        hand_ctrl: Controller instance (or None to skip).
        hand_cmd: (left_open, left_dist, right_open, right_dist) or None.
        last_left: previous left-hand open state.
        last_right: previous right-hand open state.

    Returns:
        Updated (last_left, last_right).
    """
    # Only the Dex3 controller exposes the open/close pose interface used here.
    # For other hand types (e.g. BrainCo) the dedicated deploy script drives the
    # hand with its own continuous retargeting, so this is a safe no-op.
    if hand_ctrl is None or hand_cmd is None or not hasattr(hand_ctrl, "ctrl_dual_hand"):
        return last_left, last_right

    left_open, _, right_open, _ = hand_cmd
    left_open = bool(left_open)
    right_open = bool(right_open)

    if left_open != last_left or right_open != last_right:
        left_pose = HAND_POSES["left"]["open" if left_open else "close"]
        right_pose = HAND_POSES["right"]["open" if right_open else "close"]
        hand_ctrl.ctrl_dual_hand(left_pose, right_pose)

    return left_open, right_open


# Dedicated deploy entry-point for a hand the generic script can't drive itself.
_DEPLOY_HINT = {
    "brainco": "python -m deploy.brainco.play_track_brainco",
}


def make_hand_controller(
    net: str | None = None,
    *,
    fps: int = 100,
    re_init: bool = False,
    supported: tuple[str, ...] = ("dex3", "brainco"),
):
    """Instantiate the real-robot hand controller selected by the ``HAND`` env var.

    Returns the controller, or ``None`` when no hand is mounted
    (``HAND=None`` / ``Default``), when the caller does not support the mounted
    hand (a hint to the dedicated script is logged), or when construction fails.

    ``supported`` lists the controller kinds the caller can actually command, so
    a mounted-but-unsupported hand is skipped instead of starting a controller
    the caller would never drive (whose blocking DDS init could also hang).
    Hardware SDK imports are deferred to the chosen branch.
    """
    kind = HAND_INFO[HAND]["controller"]
    if kind is None:
        return None
    if kind not in supported:
        hint = _DEPLOY_HINT.get(kind)
        print(f"[Hand] HAND={HAND} is not driven here." + (f" Use `{hint}`." if hint else ""))
        return None
    try:
        if kind == "dex3":
            return Dex3Controller(net=net, re_init=re_init)
        if kind == "brainco":
            from deploy.brainco.brainco_controller import BraincoController

            return BraincoController(fps=fps)
    except Exception as e:
        print(f"[Hand] Failed to init {kind} controller: {e}")
    return None
