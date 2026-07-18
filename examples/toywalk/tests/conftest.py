import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig, ScriptedPolicy

    from toywalk.fast import ToywalkBatched
    from toywalk.reference import ToywalkReference

    return HarnessConfig(
        name="toywalk",
        root=ENV_ROOT,
        reference_factory=ToywalkReference,
        batched_factory=lambda n, debug=False: ToywalkBatched(n, debug=debug),
        # Training-shape variant: same env, torch.compile'd step core, no
        # per-step terminal-state JSON (training bootstraps from tensors).
        # The battery bit-checks it against the eager batched env.
        benchmark_factory=lambda n, debug=False: ToywalkBatched(
            n, debug=debug, compile=True, emit_final_states=False),
        action_sampler=DiscreteActionSampler(n_actions=2),
        benchmark_batches=(1, 64, 1024, 8192),
        scripted_policies=[
            # Always-right: E[steps to goal] ~ (L - E[start]) / E[delta]
            # = 5 / 0.8 = 6.25; return = 11 - steps => ~4.75.
            ScriptedPolicy(
                name="always_right",
                policy=lambda state, t: 1,
                expected_return=4.75, tol=1.5, episodes=50),
            # Always-left: never reaches the goal, terminates at the step cap
            # with MAX_STEPS * STEP_REWARD (slips cannot rescue it from the
            # start range in expectation; verified over the fixed seeds).
            ScriptedPolicy(
                name="always_left",
                policy=lambda state, t: 0,
                expected_return=-60.0, tol=0.0, episodes=50),
        ],
        # Vectorization-win gate. toywalk's pure-Python reference is an
        # atypically fast ~1us/step (the env is ~20 lines and the counter RNG
        # is 2 integer mixes), which deflates this ratio; the compiled batched
        # env runs at tens of millions of steps/sec absolute. 20x is the
        # robust CI gate on CPU; envs with realistic reference costs clear
        # 100x easily.
        min_speedup=20.0,
    )
