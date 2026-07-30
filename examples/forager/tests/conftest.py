import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig, ScriptedPolicy

    from forager.fast import ForagerBatched
    from forager.reference import ForagerReference

    def always_north(state, t):
        return 0

    def greedy_nearest(state, t):
        """Walk toward the nearest live berry (Manhattan), x first."""
        px, py = state["pos"]
        best = None
        for cell, alive in zip(state["berries"], state["alive"]):
            if not alive:
                continue
            d = abs(cell[0] - px) + abs(cell[1] - py)
            if best is None or d < best[0]:
                best = (d, cell)
        if best is None:
            return 0
        cx, cy = best[1]
        if cx > px:
            return 1
        if cx < px:
            return 3
        if cy > py:
            return 0
        return 2

    return HarnessConfig(
        name="forager",
        root=ENV_ROOT,
        reference_factory=ForagerReference,
        batched_factory=lambda n, debug=False: ForagerBatched(n, debug=debug),
        # Training-shape variant: same env, torch.compile'd step core, no
        # per-step terminal-state JSON. The battery bit-checks it against the
        # eager batched env.
        benchmark_factory=lambda n, debug=False: ForagerBatched(
            n, debug=debug, compile=True, emit_final_states=False),
        action_sampler=DiscreteActionSampler(n_actions=4),
        benchmark_batches=(1, 64, 1024, 8192),
        # Both baselines are measured, not derived: the reference env is
        # deterministic given the fixed seed block, so these means are exactly
        # reproducible and act as a regression gate on the reward function.
        # The 13.6-point gap between them is the real assertion: a policy that
        # seeks berries must beat one that ignores them by roughly that much.
        scripted_policies=[
            ScriptedPolicy(name="greedy_nearest", policy=greedy_nearest,
                           expected_return=9.196, tol=0.05, episodes=50),
            ScriptedPolicy(name="always_north", policy=always_north,
                           expected_return=-4.404, tol=0.05, episodes=50),
        ],
        # Vectorization-win gate. Measured at 13-18x across runs on one
        # container (compiled, n=8192). The ratio is noisy because it divides
        # by a reference impl that is itself fast (~170k steps/s), so judge the
        # absolute number too (~2.3-2.9M steps/s). 8x is the robust CI floor.
        min_speedup=8.0,
    )
