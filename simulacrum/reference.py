"""ReferenceEnv: the slow, readable, single-instance implementation.

A reference implementation exists to be *obviously correct against spec.md*.
Style rules (enforced socially, not mechanically):

- One environment instance. No batching, no tensors, no vectorization.
- Dataclass state. Explicit ``if``/``for`` control flow. No premature
  abstraction — every rule in ``step`` should be traceable to a spec.md line.
- All randomness goes through ``simulacrum.rng`` scalar draws using
  ``self.key`` and the current step counter, with slots enumerated in spec.md.
  Never ``random``/``numpy.random``/``torch.rand`` — those cannot be made
  bit-identical to the batched implementation.
- ``reset(seed, episode)``: the harness replays multi-episode differential
  runs by calling reset with an incrementing ``episode``; the batched
  implementation's auto-reset derives the same episode keys internally.

The reference env does NOT auto-reset. The caller observes ``terminated`` and
calls ``reset`` explicitly. (The batched env auto-resets in-tensor; the
harness bridges the two conventions.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from simulacrum import rng


class ReferenceEnv(ABC):
    """Single-instance reference implementation, written from spec.md only.

    Subclasses must set ``self.key`` (via :meth:`seed_episode`) in ``reset``
    and are expected to track their own step counter in state for RNG draws.
    """

    def seed_episode(self, seed: int, episode: int) -> int:
        """Derive and store the episode key. Call at the top of ``reset``."""
        self.key = rng.episode_key(seed, episode)
        return self.key

    @abstractmethod
    def reset(self, seed: int, episode: int = 0) -> Any:
        """Start episode ``episode`` of the stream defined by ``seed``.

        Returns the initial state (a dataclass instance). Must call
        ``self.seed_episode(seed, episode)`` and be fully deterministic given
        ``(seed, episode)``.
        """

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Advance one step. Returns ``(state, reward, terminated, info)``.

        Must not be called on a terminated episode (call ``reset`` first).
        """

    @abstractmethod
    def observe(self, state: Any) -> Any:
        """Map a state to the observation the agent sees (numpy array or
        plain Python structure — NOT torch tensors)."""

    @abstractmethod
    def to_json(self, state: Any) -> dict:
        """Serialize a state to a dict conforming to schema.json's ``state``
        definition. Keys and dtypes must match the schema exactly — this is
        the representation the differential test compares."""

    @abstractmethod
    def from_json(self, obj: dict) -> Any:
        """Inverse of :meth:`to_json`. Used for trajectory replay."""
