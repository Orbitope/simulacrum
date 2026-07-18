"""The validation battery. Environment packages get all of it with two lines:

tests/conftest.py::

    pytest_plugins = ["simulacrum.harness.plugin"]

    @pytest.fixture
    def harness_config():
        return HarnessConfig(...)

tests/test_battery.py::

    from simulacrum.harness.battery import *

An environment that has not passed this battery is not eligible for training
(see ``simulacrum.harness.report.require_fresh_report``).
"""

from __future__ import annotations

import time

import pytest
import torch

from simulacrum import spec as spec_mod
from simulacrum.harness.differential import run_differential
from simulacrum.harness.diffs import diff_states, format_diffs
from simulacrum.traj.reader import read_trajectory
from simulacrum.traj.writer import TrajectoryWriter

__all__ = [
    "test_spec_contract",
    "test_differential",
    "test_batch_independence",
    "test_invariant_sweep",
    "test_auto_reset",
    "test_determinism",
    "test_replay",
    "test_scripted_policies",
    "test_benchmark_factory_parity",
    "test_throughput",
]

_INDEPENDENCE_SEED_BASE = 5000


def _solo_trajectory(cfg, seed: int, n_steps: int):
    """Run one instance alone (batched n=1); returns per-step records."""
    env = cfg.batched_factory(n=1, debug=False)
    env.reset(torch.tensor([seed], dtype=torch.int64))
    records = []
    for t in range(n_steps):
        a = cfg.action_sampler.to_tensor([cfg.action_sampler.sample(seed, t)])
        obs, reward, term, info = env.step(a)
        records.append({
            "state": env.slice_to_json(0),
            "obs": obs[0].clone(),
            "reward": float(reward[0]),
            "terminated": bool(term[0]),
            "final_state": info.get("final_state_json", {}).get(0),
        })
    return records


def test_spec_contract(harness_config):
    """spec.md exists with all required sections and no TODOs; schema.json is
    a valid JSON Schema with state/action/trajectory definitions."""
    spec_mod.load_spec(harness_config.root)
    spec_mod.load_schema(harness_config.root)


def test_differential(harness_config, schema_atol, record_extra):
    """K seeds x T steps: reference and batched(1) must produce identical
    states, observations, rewards, and termination flags at every step."""
    cfg = harness_config
    tol_used = {}
    for k in range(cfg.n_seeds):
        result = run_differential(cfg, seed=k, atol=schema_atol,
                                  dump_dir=cfg.root / "failures" / "differential")
        assert result.ok, f"[seed {k}] {result.divergence}"
        tol_used.update(result.tolerance_fields_used)
    record_extra("tolerance_fields_used", tol_used)


def test_batch_independence(harness_config):
    """Instance i inside a batch of N must be bit-identical to instance i run
    alone. Catches cross-instance leakage from broadcasting bugs."""
    cfg = harness_config
    n = cfg.independence_batch
    seeds = [_INDEPENDENCE_SEED_BASE + i for i in range(n)]
    n_steps = cfg.n_steps

    solos = [_solo_trajectory(cfg, s, n_steps) for s in seeds]

    env = cfg.batched_factory(n=n, debug=False)
    env.reset(torch.tensor(seeds, dtype=torch.int64))
    for t in range(n_steps):
        actions = cfg.action_sampler.to_tensor(
            [cfg.action_sampler.sample(s, t) for s in seeds])
        obs, reward, term, info = env.step(actions)
        for i in range(n):
            solo = solos[i][t]
            diffs = diff_states(solo["state"], env.slice_to_json(i), {})
            assert not diffs, (
                f"instance {i} (seed {seeds[i]}) diverged from its solo run at step {t}"
                f" inside a batch of {n} — cross-instance leakage:\n{format_diffs(diffs)}")
            assert bool(term[i]) == solo["terminated"], (
                f"instance {i} termination diverged at step {t} in-batch vs solo")
            assert float(reward[i]) == solo["reward"], (
                f"instance {i} reward diverged at step {t} in-batch vs solo:"
                f" solo={solo['reward']!r} batch={float(reward[i])!r}")
            assert torch.equal(obs[i], solo["obs"]), (
                f"instance {i} observation diverged at step {t} in-batch vs solo")


def test_invariant_sweep(harness_config):
    """Long random-action rollouts across a large batch with debug=True: every
    registered invariant checked over the whole batch after every step."""
    cfg = harness_config
    env = cfg.batched_factory(n=cfg.sweep_batch, debug=True)
    env.failures_dir = cfg.root / "failures"
    assert env._invariants, (
        "no invariants registered on the batched env — spec.md requires an "
        "enumerated invariant list; register them with @simulacrum.invariant")
    seeds = list(range(10_000, 10_000 + cfg.sweep_batch))
    env.reset(torch.tensor(seeds, dtype=torch.int64))
    for t in range(cfg.sweep_steps):
        actions = cfg.action_sampler.to_tensor(
            [cfg.action_sampler.sample(s, t) for s in seeds])
        env.step(actions)  # raises InvariantViolation (with snapshot) on failure


def test_auto_reset(harness_config, schema_atol):
    """At the first mid-batch termination: the reset instance exactly matches
    a fresh reference reset of the next episode; the returned observation is
    the first of the next episode; untouched neighbors stay bit-identical to
    their solo runs across the boundary."""
    cfg = harness_config
    n = 4
    seeds = [_INDEPENDENCE_SEED_BASE + i for i in range(n)]
    cap = cfg.n_steps * 10

    solos = [_solo_trajectory(cfg, s, cap) for s in seeds]

    env = cfg.batched_factory(n=n, debug=False)
    env.reset(torch.tensor(seeds, dtype=torch.int64))
    episodes = [0] * n
    ref = cfg.reference_factory()

    saw_termination = False
    for t in range(cap):
        actions = cfg.action_sampler.to_tensor(
            [cfg.action_sampler.sample(s, t) for s in seeds])
        obs, reward, term, info = env.step(actions)
        if not bool(term.any()):
            continue
        saw_termination = True
        for i in torch.nonzero(term, as_tuple=False).flatten().tolist():
            # Terminal state must be exposed.
            assert i in info.get("final_state_json", {}), (
                f"instance {i} terminated at step {t} but info['final_state_json'] "
                f"has no entry for it")
            episodes[i] += 1
            # Reset instance == fresh reference reset of the *next* episode.
            fresh = ref.reset(seeds[i], episodes[i])
            diffs = diff_states(ref.to_json(fresh), env.slice_to_json(i), schema_atol)
            assert not diffs, (
                f"auto-reset state of instance {i} (step {t}, episode {episodes[i]}) "
                f"does not match a fresh reset:\n{format_diffs(diffs)}")
            # Observation-emission ordering: the obs returned at the terminal
            # step is the FIRST observation of the next episode.
            assert torch.equal(obs[i], env.observe()[i]), (
                f"obs returned for terminated instance {i} at step {t} is not the "
                f"first observation of the next episode (reset-before-observe "
                f"ordering violated)")
        # Neighbors bit-untouched by the reset.
        for j in range(n):
            if bool(term[j]):
                continue
            diffs = diff_states(solos[j][t]["state"], env.slice_to_json(j), {})
            assert not diffs, (
                f"non-terminated neighbor {j} was perturbed by instance reset at "
                f"step {t}:\n{format_diffs(diffs)}")
        if all(e > 0 for e in episodes):
            break
    if not saw_termination:
        pytest.skip(
            f"no episode terminated within {cap} random-action steps; auto-reset "
            f"path not exercised (make episodes terminate under random actions, "
            f"or raise n_steps)")


def test_determinism(harness_config):
    """Same seeds + same actions twice -> bit-identical trajectories."""
    cfg = harness_config
    runs = []
    for _ in range(2):
        env = cfg.batched_factory(n=4, debug=False)
        seeds = [7000 + i for i in range(4)]
        env.reset(torch.tensor(seeds, dtype=torch.int64))
        states = []
        for t in range(cfg.n_steps):
            actions = cfg.action_sampler.to_tensor(
                [cfg.action_sampler.sample(s, t) for s in seeds])
            env.step(actions)
            states.append([env.slice_to_json(i) for i in range(4)])
        runs.append(states)
    for t, (a, b) in enumerate(zip(*runs)):
        for i, (sa, sb) in enumerate(zip(a, b)):
            diffs = diff_states(sa, sb, {})
            assert not diffs, (
                f"two identical runs diverged at step {t}, instance {i} — "
                f"nondeterminism:\n{format_diffs(diffs)}")


def test_replay(harness_config, tmp_path):
    """A dumped reference trajectory, replayed through a fresh reference env,
    reproduces itself (also round-trips the writer/reader + schema)."""
    cfg = harness_config
    schema = spec_mod.load_schema(cfg.root)
    seed = 42
    ref = cfg.reference_factory()
    episode = 0
    state = ref.reset(seed, episode)
    writer = TrajectoryWriter(cfg.name, seed, ref.to_json(state), env_root=cfg.root)
    for t in range(cfg.n_steps):
        action = cfg.action_sampler.sample(seed, t)
        state, reward, term, _ = ref.step(action)
        writer.record(t, episode, ref.to_json(state), action, reward, term)
        if term:
            episode += 1
            state = ref.reset(seed, episode)
    path = writer.write(tmp_path / "replay.json")

    traj = read_trajectory(path, schema=schema)  # validates against schema

    ref2 = cfg.reference_factory()
    episode = 0
    state = ref2.reset(traj.metadata["seed"], episode)
    diffs = diff_states(traj.initial["state"], ref2.to_json(state), {})
    assert not diffs, f"replay initial state mismatch:\n{format_diffs(diffs)}"
    for step in traj.steps:
        state, reward, term, _ = ref2.step(step["action"])
        diffs = diff_states(step["state"], ref2.to_json(state), {})
        assert not diffs, (
            f"replay diverged at step {step['t']}:\n{format_diffs(diffs)}")
        assert float(reward) == step["reward"] and bool(term) == step["terminated"]
        if term:
            episode += 1
            state = ref2.reset(traj.metadata["seed"], episode)


def test_scripted_policies(harness_config):
    """Author-supplied (policy, expected_return, tolerance) cases, run on the
    reference implementation."""
    cfg = harness_config
    if not cfg.scripted_policies:
        pytest.skip("no scripted policies provided (HarnessConfig.scripted_policies)")
    ref = cfg.reference_factory()
    for sp in cfg.scripted_policies:
        returns = []
        for e in range(sp.episodes):
            seed = sp.base_seed + e
            state = ref.reset(seed, 0)
            total, t = 0.0, 0
            while t < sp.max_steps:
                action = sp.policy(ref.to_json(state), t)
                state, reward, term, _ = ref.step(action)
                total += reward
                t += 1
                if term:
                    break
            returns.append(total)
        mean = sum(returns) / len(returns)
        assert abs(mean - sp.expected_return) <= sp.tol, (
            f"scripted policy '{sp.name}': mean return {mean:.4f} over "
            f"{sp.episodes} episodes, expected {sp.expected_return} +/- {sp.tol}")


def test_benchmark_factory_parity(harness_config):
    """If the env supplies a separate benchmark/training factory (e.g. a
    torch.compile'd variant), it must be bit-identical to the plain batched
    factory — otherwise the differential guarantee doesn't transfer to the
    thing that actually trains."""
    cfg = harness_config
    if cfg.benchmark_factory is None:
        pytest.skip("no separate benchmark_factory configured")
    n = 16
    seeds = torch.tensor([9000 + i for i in range(n)], dtype=torch.int64)
    plain = cfg.batched_factory(n=n, debug=False)
    bench = cfg.benchmark_factory(n=n, debug=False)
    plain.reset(seeds.clone())
    bench.reset(seeds.clone())
    for t in range(cfg.n_steps):
        actions = cfg.action_sampler.to_tensor(
            [cfg.action_sampler.sample(int(s), t) for s in seeds])
        obs_p, rew_p, term_p, _ = plain.step(actions)
        obs_b, rew_b, term_b, _ = bench.step(actions.clone())
        assert torch.equal(term_p, term_b), f"terminated diverged at step {t}"
        assert torch.equal(rew_p, rew_b), f"rewards diverged at step {t}"
        assert torch.equal(obs_p.view(-1), obs_b.view(-1)), f"obs diverged at step {t}"
        for i in range(n):
            diffs = diff_states(plain.slice_to_json(i), bench.slice_to_json(i), {})
            assert not diffs, (
                f"benchmark factory state diverged from plain batched at step {t}, "
                f"instance {i}:\n{format_diffs(diffs)}")


def test_throughput(harness_config, record_extra):
    """Steps/sec for the reference impl and the batched impl at several batch
    sizes, debug on/off. Persisted in the validation report so perf
    regressions are visible. Asserts only HarnessConfig.min_speedup (default
    0 = record-only)."""
    cfg = harness_config
    n_steps = cfg.benchmark_steps
    results: dict = {}

    ref = cfg.reference_factory()
    seed, episode = 0, 0
    state = ref.reset(seed, episode)
    actions = [cfg.action_sampler.sample(seed, t) for t in range(n_steps)]
    t0 = time.perf_counter()
    for t in range(n_steps):
        state, _, term, _ = ref.step(actions[t])
        if term:
            episode += 1
            state = ref.reset(seed, episode)
    ref_sps = n_steps / (time.perf_counter() - t0)
    results["reference_steps_per_sec"] = ref_sps

    factory = cfg.benchmark_factory or cfg.batched_factory
    results["batched"] = {}
    for n in cfg.benchmark_batches:
        seeds = list(range(n))
        action_tensors = [
            cfg.action_sampler.to_tensor([cfg.action_sampler.sample(s, t) for s in seeds])
            for t in range(n_steps)
        ]
        for debug in (False, True):
            # debug runs always use the plain factory (debug forces eager).
            env = (cfg.batched_factory if debug else factory)(n=n, debug=debug)
            env.reset(torch.tensor(seeds, dtype=torch.int64))
            env.step(action_tensors[0])  # warmup (JIT/compile, allocations)
            env.reset(torch.tensor(seeds, dtype=torch.int64))
            t0 = time.perf_counter()
            for t in range(n_steps):
                env.step(action_tensors[t])
            sps = n * n_steps / (time.perf_counter() - t0)
            results["batched"][f"n={n},debug={debug}"] = sps

    largest = max(cfg.benchmark_batches)
    speedup = results["batched"][f"n={largest},debug=False"] / ref_sps
    results["speedup_at_largest_batch"] = speedup
    record_extra("throughput", results)
    if cfg.min_speedup > 0:
        assert speedup >= cfg.min_speedup, (
            f"batched throughput at n={largest} is only {speedup:.1f}x the "
            f"reference (required: {cfg.min_speedup}x)")
