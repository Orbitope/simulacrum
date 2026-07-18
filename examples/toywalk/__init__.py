"""toywalk: the deliberately boring simulacrum example environment."""

from enum import IntEnum

L = 5
MAX_STEPS = 60
SLIP_P = 0.1
START_RANGE = 2
GOAL_REWARD = 10.0
STEP_REWARD = -1.0


class Slots(IntEnum):
    """RNG slots — mirrors the table in spec.md."""
    INIT_POSITION = 0   # reset-time (step 0)
    SLIP = 1            # per-step
