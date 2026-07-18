"""Batched tensor implementation of toywalk.

Written from spec.md ONLY — not from reference.py. State: int64 tensors with
leading [N] dim; branching replaced by torch.where masking; the base class
handles auto-reset and RNG keying (self.keys, self.t).
"""

from __future__ import annotations

import torch

from simulacrum import BatchedEnv, invariant, rng

from toywalk import (
    GOAL_REWARD, L, MAX_STEPS, SLIP_P, START_RANGE, STEP_REWARD, Slots,
)


class ToywalkBatched(BatchedEnv):
    def _reset_instances(self, mask: torch.Tensor) -> None:
        # spec: Reset — uniform over [-START_RANGE, +START_RANGE], slot
        # INIT_POSITION at step 0. Draw for ALL instances; stateless RNG makes
        # the discarded (unmasked) draws harmless.
        init_pos = rng.draw_randint_torch(
            self.keys, 0, Slots.INIT_POSITION, 2 * START_RANGE + 1) - START_RANGE
        if not hasattr(self, "position"):
            self.position = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.position = torch.where(mask, init_pos, self.position)
        # self.t is zeroed for masked instances by the base class.

    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # spec: Actions — 0 = left (-1), 1 = right (+1).
        delta = torch.where(actions == 1,
                            torch.ones_like(self.position),
                            -torch.ones_like(self.position))

        # spec: RNG slots — SLIP keyed on the pre-move step counter.
        slip = rng.draw_bernoulli_torch(self.keys, self.t, Slots.SLIP, SLIP_P)
        delta = torch.where(slip, -delta, delta)

        # spec: transition — clamp(position + delta, -L, L).
        self.position = torch.clamp(self.position + delta, -L, L)

        at_goal = self.position == L

        # spec: Rewards — +10 on landing on the goal, else -1.
        rewards = torch.where(
            at_goal,
            torch.full_like(self.position, GOAL_REWARD, dtype=torch.float64),
            torch.full_like(self.position, STEP_REWARD, dtype=torch.float64))

        # spec: Termination — goal reached, or step cap hit (post-move t).
        terminated = at_goal | (self.t + 1 == MAX_STEPS)
        return rewards, terminated

    def observe(self) -> torch.Tensor:
        # spec: Observations — float32, divisions computed in float32.
        return torch.stack(
            [self.position.to(torch.float32) / float(L),
             self.t.to(torch.float32) / float(MAX_STEPS)],
            dim=-1)

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {"position": self.position, "t": self.t}

    # spec: Invariants 1-3.

    @invariant("in_bounds")
    def _inv_in_bounds(self) -> torch.Tensor:
        return (self.position >= -L) & (self.position <= L)

    @invariant("step_counter")
    def _inv_step_counter(self) -> torch.Tensor:
        return (self.t >= 0) & (self.t <= MAX_STEPS)

    @invariant("init_in_start_range")
    def _inv_init_in_start_range(self) -> torch.Tensor:
        return (self.t != 0) | (self.position.abs() <= START_RANGE)
