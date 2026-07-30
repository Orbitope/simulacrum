"""Batched tensor implementation of forager.

Written from spec.md ONLY, not from reference.py. State: tensors with leading
[N] dim; branching replaced by torch.where masking; the base class handles
auto-reset and RNG keying (self.keys, self.t).

Note the unsqueeze discipline throughout. State here is *shaped*: [N, 2]
positions, [N, K] flags, [N, K, 2] berry cells, so a [N] mask must be lifted
to [N, 1] or [N, 1, 1] before it meets state. A bare [N] mask against [N, 1]
broadcasts to [N, N]: correct at n=1, cross-contaminated above it, and
invisible to a differential test that only ever runs one instance. That is
what test_batch_independence exists to catch.
"""

from __future__ import annotations

import torch

from simulacrum import BatchedEnv, invariant, rng

from forager import (
    DELTAS, ENERGY_MAX, G, GAINS, GUST_P, K, MAX_STEPS, N_KINDS, REWARD_SCALE,
    REWARD_STEP, START_ENERGY, STEP_COST, Slots,
)


class ForagerBatched(BatchedEnv):
    def _alloc(self) -> None:
        """Allocate state tensors and the lookup constants, once."""
        n, dev = self.n, self.device
        self.pos = torch.zeros(n, 2, dtype=torch.int64, device=dev)
        self.berries = torch.zeros(n, K, 2, dtype=torch.int64, device=dev)
        self.kinds = torch.zeros(n, K, dtype=torch.int64, device=dev)
        self.alive = torch.ones(n, K, dtype=torch.bool, device=dev)
        self.energy = torch.zeros(n, dtype=torch.float32, device=dev)
        # spec: Actions. Delta lookup, gathered by action index.
        self._deltas = torch.tensor(DELTAS, dtype=torch.int64, device=dev)
        # spec: Collection. Gains indexed by berry kind, summed in float64.
        self._gains = torch.tensor([float(g) for g in GAINS],
                                   dtype=torch.float64, device=dev)

    def _reset_instances(self, mask: torch.Tensor) -> None:
        if not hasattr(self, "pos"):
            self._alloc()

        # spec: Reset. Agent cell, slots AGENT_X / AGENT_Y at step 0, index 0.
        # Draw for ALL instances; the stateless RNG makes discarded draws safe.
        ax = rng.draw_randint_torch(self.keys, 0, Slots.AGENT_X, G)
        ay = rng.draw_randint_torch(self.keys, 0, Slots.AGENT_Y, G)
        pos = torch.stack([ax, ay], dim=-1)                       # [N, 2]

        # spec: Reset. Berry k drawn at index = k. The index word varies along
        # the BERRY axis, never the batch axis; K is a constant, so this loop
        # unrolls at trace time and stays compilable.
        bx = torch.stack(
            [rng.draw_randint_torch(self.keys, 0, Slots.BERRY_X, G, index=k)
             for k in range(K)], dim=-1)                          # [N, K]
        by = torch.stack(
            [rng.draw_randint_torch(self.keys, 0, Slots.BERRY_Y, G, index=k)
             for k in range(K)], dim=-1)                          # [N, K]
        berries = torch.stack([bx, by], dim=-1)                   # [N, K, 2]
        kinds = torch.stack(
            [rng.draw_randint_torch(self.keys, 0, Slots.BERRY_KIND, N_KINDS, index=k)
             for k in range(K)], dim=-1)                          # [N, K]

        # Masked fill. Each mask is lifted to the rank of the state it selects.
        m1 = mask.unsqueeze(-1)                                   # [N, 1]
        self.pos = torch.where(m1, pos, self.pos)
        self.berries = torch.where(m1.unsqueeze(-1), berries, self.berries)
        self.kinds = torch.where(m1, kinds, self.kinds)
        self.alive = torch.where(m1, torch.ones_like(self.alive), self.alive)
        self.energy = torch.where(
            mask, torch.full_like(self.energy, float(START_ENERGY)), self.energy)
        # self.t is zeroed for masked instances by the base class.

    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # spec: Actions. The intended delta for this compass direction.
        delta = self._deltas[actions]                             # [N, 2]

        # spec: transition step 2. GUST keyed on the PRE-move step counter;
        # if True the delta rotates 90 degrees clockwise: (dx, dy) -> (dy, -dx).
        gust = rng.draw_bernoulli_torch(self.keys, self.t, Slots.GUST, GUST_P)
        rotated = torch.stack([delta[:, 1], -delta[:, 0]], dim=-1)
        delta = torch.where(gust.unsqueeze(-1), rotated, delta)

        # spec: transition step 3. Clamp AFTER the rotation.
        self.pos = torch.clamp(self.pos + delta, 0, G - 1)

        # spec: Collection. A live berry on the post-move cell is collected.
        # berries [N, K, 2] against pos lifted to [N, 1, 2] -> [N, K].
        on_cell = (self.berries == self.pos.unsqueeze(1)).all(-1)  # [N, K]
        collected = on_cell & self.alive                           # [N, K]
        self.alive = self.alive & ~collected

        # spec: Collection. Gains summed in ascending k order. Masked-out
        # berries contribute an exact 0.0, so the reduction matches a scalar
        # accumulation that skips them.
        gained = (self._gains[self.kinds] * collected.to(torch.float64)).sum(-1)

        # spec: Energy. Float32, left to right: subtract cost, then add gain.
        self.energy = (self.energy - float(STEP_COST)) + gained.to(torch.float32)

        # spec: Rewards. Float64, in this order.
        rewards = REWARD_SCALE * gained + REWARD_STEP

        # spec: Termination. All collected, energy exhausted, or step cap.
        alive_count = self.alive.sum(-1)
        terminated = ((alive_count == 0)
                      | (self.energy <= 0.0)
                      | (self.t + 1 == MAX_STEPS))
        return rewards, terminated

    def observe(self) -> torch.Tensor:
        # spec: Observations. Float32[5], every division computed in float32.
        alive_count = self.alive.sum(-1)
        return torch.stack(
            [self.pos[:, 0].to(torch.float32) / float(G - 1),
             self.pos[:, 1].to(torch.float32) / float(G - 1),
             self.energy,
             alive_count.to(torch.float32) / float(K),
             self.t.to(torch.float32) / float(MAX_STEPS)],
            dim=-1)

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {"pos": self.pos, "berries": self.berries, "kinds": self.kinds,
                "alive": self.alive, "energy": self.energy, "t": self.t}

    # spec: Invariants 1-6.

    @invariant("pos_in_bounds")
    def _inv_pos(self) -> torch.Tensor:
        return ((self.pos >= 0) & (self.pos <= G - 1)).all(-1)

    @invariant("berries_in_bounds")
    def _inv_berries(self) -> torch.Tensor:
        return ((self.berries >= 0) & (self.berries <= G - 1)).all(-1).all(-1)

    @invariant("kinds_valid")
    def _inv_kinds(self) -> torch.Tensor:
        return ((self.kinds >= 0) & (self.kinds <= N_KINDS - 1)).all(-1)

    @invariant("step_counter")
    def _inv_step_counter(self) -> torch.Tensor:
        return (self.t >= 0) & (self.t <= MAX_STEPS)

    @invariant("energy_bounded_above")
    def _inv_energy(self) -> torch.Tensor:
        return self.energy <= ENERGY_MAX

    @invariant("fresh_episode")
    def _inv_fresh(self) -> torch.Tensor:
        return (self.t != 0) | (self.alive.all(-1)
                                & (self.energy == float(START_ENERGY)))
