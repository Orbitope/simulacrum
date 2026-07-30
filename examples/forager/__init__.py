"""forager: the simulacrum example environment with *shaped* state.

toywalk is the minimal case — two scalar fields. forager is the one that
exercises the idioms a real environment needs: vector state (`[K, 2]` berry
positions, `[K]` flags), per-item RNG draws via the `index` word, and
collection through a masked scatter.
"""

from enum import IntEnum

import numpy as np

G = 8                  # grid is G x G, coordinates 0..G-1
K = 6                  # berries per episode
N_KINDS = 3
MAX_STEPS = 80
GUST_P = 0.15

# float32 constants — energy arithmetic is performed in float32 (spec #Energy).
START_ENERGY = np.float32(1.0)
STEP_COST = np.float32(0.02)
GAINS = (np.float32(0.1), np.float32(0.15), np.float32(0.3))

# spec: State space — energy is bounded above only.
ENERGY_MAX = float(START_ENERGY) + K * float(max(GAINS))

# float64 reward constants.
REWARD_SCALE = 10.0
REWARD_STEP = -0.1

# spec: Actions — 0 = north, 1 = east, 2 = south, 3 = west.
DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class Slots(IntEnum):
    """RNG slots — mirrors the table in spec.md."""
    AGENT_X = 0      # reset (step 0), index 0
    AGENT_Y = 1      # reset (step 0), index 0
    BERRY_X = 2      # reset (step 0), index k
    BERRY_Y = 3      # reset (step 0), index k
    BERRY_KIND = 4   # reset (step 0), index k
    GUST = 5         # per-step, index 0
