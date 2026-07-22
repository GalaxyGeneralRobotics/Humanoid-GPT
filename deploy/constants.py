"""Deploy-specific constants (mocap, walk policy, real-robot networking)."""

import numpy as np

# Mounted end-effector selection (None / Default / Dex3 / BrainCo). Re-exported
# from utils.hand so deploy code can pick the right real-robot hand controller
# without importing the training stack.
from utils.hand import HAND, HAND_INFO  # noqa: F401

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
NUM_JOINT = 29

# ---------------------------------------------------------------------------
# Unitree DDS topics
# ---------------------------------------------------------------------------
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"

# ---------------------------------------------------------------------------
# Walk policy command scaling
# ---------------------------------------------------------------------------
DEFAULT_CHEST_Z = 1.05
CMD_SCALE_VEL = [0.5, 0.3, 0.5]
KEYBOARD_MAX_SPEED = 0.35

# ---------------------------------------------------------------------------
# Walk-mode PD gains (higher stiffness for locomotion)
# ---------------------------------------------------------------------------
# fmt: off
KPs_walking = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs_walking = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])
# fmt: on

# ---------------------------------------------------------------------------
# Motor IDs  (shared with tracking/constants.py but repeated here so that
# deploy/ does not need to import the training stack for real-robot control)
# ---------------------------------------------------------------------------


class MotorID:
    LEG_L = [0, 1, 2, 3, 4, 5]
    LEG_R = [6, 7, 8, 9, 10, 11]
    WAIST = [12, 13, 14]
    ARM_L = [15, 16, 17, 18, 19, 20, 21]
    ARM_R = [22, 23, 24, 25, 26, 27, 28]
    LEGs = LEG_L + LEG_R
    ARMs = ARM_L + ARM_R
    FULL = LEG_L + LEG_R + WAIST + ARM_L + ARM_R

# ---------------------------------------------------------------------------
# Default joint configuration (joint-only, no root)
# ---------------------------------------------------------------------------
# fmt: off
DEFAULT_QPOS = np.float32([
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])
# fmt: on
