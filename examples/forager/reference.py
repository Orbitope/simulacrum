"""Readable single-instance reference implementation of forager.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulacrum import ReferenceEnv, rng

from forager import (
    DELTAS, G, GAINS, GUST_P, K, MAX_STEPS, N_KINDS, REWARD_SCALE,
    REWARD_STEP, START_ENERGY, STEP_COST, Slots,
)


@dataclass
class State:
    pos: list          # spec: State space. [x, y], each in [0, G-1]
    berries: list      # spec: State space. K cells, fixed for the episode
    kinds: list        # spec: State space. Berry kind per berry
    alive: list        # spec: State space. Berry not yet collected
    energy: np.float32  # spec: State space. Float32
    t: int             # spec: State space. In-episode step counter


class ForagerReference(ReferenceEnv):
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)

        # spec: Reset. Agent cell, slots AGENT_X / AGENT_Y at step 0, index 0.
        pos = [rng.draw_randint(self.key, 0, Slots.AGENT_X, G),
               rng.draw_randint(self.key, 0, Slots.AGENT_Y, G)]

        # spec: Reset. Berry k drawn at index = k. Overlaps are legal.
        berries = []
        kinds = []
        for k in range(K):
            x = rng.draw_randint(self.key, 0, Slots.BERRY_X, G, index=k)
            y = rng.draw_randint(self.key, 0, Slots.BERRY_Y, G, index=k)
            berries.append([x, y])
            kinds.append(rng.draw_randint(self.key, 0, Slots.BERRY_KIND, N_KINDS, index=k))

        self.state = State(pos=pos, berries=berries, kinds=kinds,
                           alive=[True] * K, energy=START_ENERGY, t=0)
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        state = self.state

        # spec: Actions. The intended delta for this compass direction.
        dx, dy = DELTAS[action]

        # spec: transition step 2. The GUST draw is keyed on the PRE-move step
        # counter; if True the delta rotates 90 degrees clockwise.
        if rng.draw_bernoulli(self.key, state.t, Slots.GUST, GUST_P):
            dx, dy = dy, -dx

        # spec: transition step 3. Clamp AFTER the rotation, per coordinate.
        x = state.pos[0] + dx
        y = state.pos[1] + dy
        if x < 0:
            x = 0
        if x > G - 1:
            x = G - 1
        if y < 0:
            y = 0
        if y > G - 1:
            y = G - 1

        # spec: transition step 4.
        t = state.t + 1

        # spec: Collection. A live berry on the post-move cell is collected;
        # gains are summed in ascending k order.
        alive = list(state.alive)
        gained = 0.0
        for k in range(K):
            if alive[k] and state.berries[k][0] == x and state.berries[k][1] == y:
                alive[k] = False
                gained += float(GAINS[state.kinds[k]])

        # spec: Energy. Float32, left to right: subtract cost, then add gain.
        energy = np.float32(state.energy - STEP_COST) + np.float32(gained)

        # spec: Rewards. Float64, in this order.
        reward = REWARD_SCALE * gained + REWARD_STEP

        # spec: Termination. All collected, energy exhausted, or step cap.
        alive_count = sum(1 for a in alive if a)
        terminated = (alive_count == 0) or (float(energy) <= 0.0) or (t == MAX_STEPS)

        self.state = State(pos=[x, y], berries=state.berries, kinds=state.kinds,
                           alive=alive, energy=energy, t=t)
        return self.state, reward, terminated, {}

    def observe(self, state: State) -> np.ndarray:
        # spec: Observations. Float32[5], every division computed in float32.
        alive_count = sum(1 for a in state.alive if a)
        return np.array(
            [np.float32(state.pos[0]) / np.float32(G - 1),
             np.float32(state.pos[1]) / np.float32(G - 1),
             np.float32(state.energy),
             np.float32(alive_count) / np.float32(K),
             np.float32(state.t) / np.float32(MAX_STEPS)],
            dtype=np.float32,
        )

    def to_json(self, state: State) -> dict:
        return {
            "pos": [int(v) for v in state.pos],
            "berries": [[int(c[0]), int(c[1])] for c in state.berries],
            "kinds": [int(v) for v in state.kinds],
            "alive": [bool(a) for a in state.alive],
            "energy": float(state.energy),
            "t": int(state.t),
        }

    def from_json(self, obj: dict) -> State:
        return State(
            pos=[int(v) for v in obj["pos"]],
            berries=[[int(c[0]), int(c[1])] for c in obj["berries"]],
            kinds=[int(v) for v in obj["kinds"]],
            alive=[bool(a) for a in obj["alive"]],
            energy=np.float32(obj["energy"]),
            t=int(obj["t"]),
        )
