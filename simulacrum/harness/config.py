"""HarnessConfig: the single object an environment package provides to the
validation battery (via a ``harness_config`` pytest fixture in its conftest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from simulacrum import rng
from simulacrum.batched import BatchedEnv
from simulacrum.reference import ReferenceEnv

# Domain tag separating the harness's action stream from env RNG streams.
_DOMAIN_ACTION = 0xAC7104AC7104AC71


class DiscreteActionSampler:
    """Deterministic random action stream for discrete action spaces.

    ``sample(seed, t)`` -> int in [0, n_actions), a pure function of the
    instance seed and global step — so the reference run, the solo batched
    run, and the in-batch run of the same seed all see the same actions.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def sample(self, seed: int, t: int) -> int:
        return rng.draw_randint(rng._hash_words(_DOMAIN_ACTION, seed), t, 0, self.n_actions)

    def to_tensor(self, actions: list[Any]) -> torch.Tensor:
        return torch.tensor(actions, dtype=torch.int64)


@dataclass
class ScriptedPolicy:
    """A policy with a known expected return, for sanity-checking rewards.

    ``policy(state_json, t) -> action``. The harness runs ``episodes``
    episodes on the reference implementation (seeds ``base_seed..base_seed+
    episodes-1``) and asserts the mean return is within ``tol`` of
    ``expected_return``.
    """
    name: str
    policy: Callable[[dict, int], Any]
    expected_return: float
    tol: float
    episodes: int = 20
    base_seed: int = 1000
    max_steps: int = 10_000


@dataclass
class HarnessConfig:
    name: str
    root: Path                                     # env package root (spec.md, schema.json live here)
    reference_factory: Callable[[], ReferenceEnv]
    batched_factory: Callable[..., BatchedEnv]     # (n: int, debug: bool = ...) -> BatchedEnv
    action_sampler: Any                            # .sample(seed, t) -> action; .to_tensor(list) -> Tensor
    # Optional training-shape factory (e.g. compile=True). Used by the
    # throughput benchmark, and bit-checked against batched_factory by the
    # compiled-parity test. Defaults to batched_factory.
    benchmark_factory: Callable[..., BatchedEnv] | None = None
    scripted_policies: list[ScriptedPolicy] = field(default_factory=list)

    # Differential test breadth: K seeds x T steps.
    n_seeds: int = 8
    n_steps: int = 300

    # Comparison tolerances outside the state schema (state-field tolerances
    # are declared per-field in schema.json via x-atol). Nonzero values are
    # recorded in the validation report.
    reward_atol: float = 0.0
    obs_atol: float = 0.0

    # Batch-independence test.
    independence_batch: int = 8
    # Invariant sweep.
    sweep_batch: int = 256
    sweep_steps: int = 500
    # Throughput benchmark.
    benchmark_batches: tuple[int, ...] = (1, 64, 1024)
    benchmark_steps: int = 300
    # If > 0, the throughput test asserts batched steps/sec at the largest
    # benchmark batch is at least this multiple of the reference impl's.
    min_speedup: float = 0.0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
