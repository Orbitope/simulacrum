"""Readable single-instance reference implementation of toywalk.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulacrum import ReferenceEnv, rng

from toywalk import (
    GOAL_REWARD, L, MAX_STEPS, SLIP_P, START_RANGE, STEP_REWARD, Slots,
)


@dataclass
class State:
    position: int  # spec: State space — location on the line, in [-L, L]
    t: int         # spec: State space — in-episode step counter


class ToywalkReference(ReferenceEnv):
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # spec: Reset — uniform over [-START_RANGE, +START_RANGE], slot
        # INIT_POSITION at step 0.
        position = rng.draw_randint(self.key, 0, Slots.INIT_POSITION,
                                    2 * START_RANGE + 1) - START_RANGE
        self.state = State(position=position, t=0)
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        state = self.state

        # spec: Actions — 0 = left (-1), 1 = right (+1).
        if action == 1:
            delta = 1
        else:
            delta = -1

        # spec: RNG slots — SLIP draw keyed on the pre-move step counter t;
        # if True, the delta is negated.
        if rng.draw_bernoulli(self.key, state.t, Slots.SLIP, SLIP_P):
            delta = -delta

        # spec: transition — clamp(position + delta, -L, L).
        position = state.position + delta
        if position > L:
            position = L
        if position < -L:
            position = -L

        t = state.t + 1

        # spec: Rewards — +10 on landing on the goal, else -1.
        if position == L:
            reward = GOAL_REWARD
        else:
            reward = STEP_REWARD

        # spec: Termination — goal reached, or step cap hit.
        terminated = (position == L) or (t == MAX_STEPS)

        self.state = State(position=position, t=t)
        return self.state, reward, terminated, {}

    def observe(self, state: State) -> np.ndarray:
        # spec: Observations — float32, divisions computed in float32.
        return np.array(
            [np.float32(state.position) / np.float32(L),
             np.float32(state.t) / np.float32(MAX_STEPS)],
            dtype=np.float32,
        )

    def to_json(self, state: State) -> dict:
        return {"position": int(state.position), "t": int(state.t)}

    def from_json(self, obj: dict) -> State:
        return State(position=int(obj["position"]), t=int(obj["t"]))
