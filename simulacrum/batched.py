"""BatchedEnv: the fast, batched tensor implementation.

VECTORIZATION IDIOMS (the contract for subclasses)
==================================================
All state lives in tensors with a leading batch dimension ``[N, ...]``. There
is no per-instance Python object, and the step path must contain no
data-dependent Python branching on instance state. The core idioms:

**Branch -> mask.** Anywhere the reference implementation writes::

    if condition:
        x = a
    else:
        x = b

the batched implementation writes::

    x = torch.where(condition, a, b)      # condition: Bool[N]

``torch.where`` evaluates BOTH branches for every instance. Consequences:
- Both branches must be safe to compute everywhere (no out-of-bounds indexing,
  no division by zero, no NaN-producing ops on the not-taken side). Clamp or
  sanitize indices *before* gathering, then mask the result.
- Discarded random draws are harmless: the framework RNG is stateless
  (counter-based), so drawing for all N instances and masking does not
  perturb any stream. Always draw for the full batch.

**Conditional update.** ``state = torch.where(mask, new_value, state)`` —
never ``state[mask] = new_value[mask]`` inside ``torch.where``-style logic
unless you need scatter semantics; if you do use boolean indexing, beware
in-place ops on views (clone first when a tensor aliases another).

**Broadcasting hygiene.** A ``[N]`` mask silently broadcasts against
``[N, D]`` state along the WRONG axis unless unsqueezed:
``torch.where(mask.unsqueeze(-1), new, old)``. Accidental ``[N] vs [N, 1]``
broadcasts produce ``[N, N]`` tensors — the classic cross-instance leak, and
exactly what the batch-independence test exists to catch.

**Dtype discipline.** Match schema.json dtypes exactly (int64 for discrete
state, float32/float64 as declared). Dtype drift (e.g. an int32 intermediate)
breaks bit-identity with the reference implementation.

AUTO-RESET CONTRACT
===================
``step`` runs a fully tensorized, branch-free core (compilable with
``compile=True``):

1. ``_step_impl(actions)`` -> ``(rewards, terminated)``, mutating state
   tensors for ALL instances (masked as needed); then the in-episode step
   counter ``self.t`` increments.
2. The post-transition state is snapshotted (tensor clones — this is where
   terminal states are captured).
3. For ALL instances, next-episode keys are computed and where-selected onto
   terminated instances; their episode counter increments and ``self.t``
   zeroes. ``_reset_instances(terminated)`` re-initializes their state
   tensors (masked — non-terminated neighbors must be bit-untouched; the
   auto-reset test enforces this).
4. ``observe()`` runs on the post-reset state, so the observation returned
   for a terminated instance is the FIRST observation of its next episode.
   The terminal state is delivered in ``info["final_state_json"]``
   (dict: index -> state JSON), built from the snapshot.

With ``debug=True`` every registered invariant runs over the whole batch
both after the transition (so terminal states are checked) and after any
auto-reset (debug mode always uses the eager path).

RNG
===
The base class owns the seeding state: ``self.seeds [N]``, ``self.episodes
[N]``, ``self.keys [N]`` (episode keys), ``self.t [N]`` (in-episode step
index). Subclasses draw with ``simulacrum.rng.draw_*_torch(self.keys, self.t,
Slot.X)``. Initialization draws (inside ``_reset_instances``) use step 0 with
their own dedicated slots; the first transition also occurs at step 0 but uses
different slots, so streams never collide. ``self.t`` increments after the
transition and zeroes on reset — the reference implementation's state must
track the same counter for its own draws.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch

from simulacrum import rng


class InvariantViolation(AssertionError):
    def __init__(self, name: str, env_index: int, snapshot_path: Path | None, message: str = ""):
        self.invariant_name = name
        self.env_index = env_index
        self.snapshot_path = snapshot_path
        detail = f" (snapshot: {snapshot_path})" if snapshot_path else ""
        super().__init__(
            f"Invariant '{name}' violated at env index {env_index}{detail}. {message}"
        )


def invariant(name: str) -> Callable:
    """Decorator registering a method as a named invariant.

    The method takes no arguments beyond ``self`` and returns ``Bool[N]``:
    True where the invariant HOLDS. With ``debug=True`` every invariant runs
    over the whole batch after every transition; a violation raises
    :class:`InvariantViolation` naming the invariant and the first failing
    env index, and dumps that instance's JSON snapshot to the failures
    directory for later rendering.
    """
    def mark(fn: Callable) -> Callable:
        fn._invariant_name = name
        return fn
    return mark


class BatchedEnv(ABC):
    """Batched tensor implementation, written from spec.md only.

    Subclasses implement ``_reset_instances``, ``_step_impl``, ``observe``,
    and ``state_tensors`` (and may override ``slice_to_json`` for nested
    state). The base class owns the step loop, RNG seeding state, auto-reset,
    terminal-state capture, and invariant checking. See the module docstring
    for the idioms and the auto-reset contract.

    ``compile=True`` wraps the tensorized step core in ``torch.compile``
    (ignored when ``debug=True``; the harness's compiled-parity test verifies
    the compiled path is bit-identical to eager).
    """

    def __init__(self, n: int, *, debug: bool = False, compile: bool = False,
                 emit_final_states: bool = True,
                 failures_dir: str | Path = "failures",
                 device: str | torch.device = "cpu"):
        self.n = n
        self.debug = debug
        self.emit_final_states = emit_final_states
        self.failures_dir = Path(failures_dir)
        self.device = torch.device(device)
        self._invariants: list[tuple[str, Callable]] = []
        for klass in type(self).__mro__:
            for attr in vars(klass).values():
                if callable(attr) and hasattr(attr, "_invariant_name"):
                    entry = (attr._invariant_name, attr)
                    if entry[0] not in [n_ for n_, _ in self._invariants]:
                        self._invariants.append(entry)
        self._compiled = bool(compile) and not debug
        self._core = self._eager_core
        if self._compiled:
            self._core = torch.compile(self._eager_core, dynamic=False)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seeds: torch.Tensor) -> None:
        """(Re)start all instances. ``seeds``: int64 ``[N]``. Instance i
        begins episode 0 of the stream defined by ``seeds[i]``."""
        if seeds.shape != (self.n,):
            raise ValueError(f"seeds must have shape [{self.n}], got {tuple(seeds.shape)}")
        self.seeds = seeds.to(dtype=torch.int64, device=self.device)
        self.episodes = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.keys = rng.episode_key_torch(self.seeds, self.episodes)
        self.t = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self._reset_instances(torch.ones(self.n, dtype=torch.bool, device=self.device))
        if self.debug:
            self._check_invariants()

    def _eager_core(self, actions: torch.Tensor):
        """The branch-free tensor step: transition, snapshot, re-key,
        auto-reset, observe. Everything here must stay data-independent
        Python-control-flow-wise so torch.compile can capture one graph."""
        rewards, terminated = self._step_impl(actions)
        self.t = self.t + 1
        if self.debug:
            self._check_invariants()  # checks terminal states, pre-reset
        # Under torch.compile the auto-reset block must run unconditionally
        # (one branch-free graph); in eager mode, skip the snapshot clones and
        # the batch-wide episode re-key on the (common) steps where nothing
        # terminated. `self._compiled` short-circuits before the tensor sync.
        any_terminated = self._compiled or bool(terminated.any())
        snapshot = None
        if self.emit_final_states and any_terminated:
            snapshot = {name: tensor.clone()
                        for name, tensor in self.state_tensors().items()}
        if any_terminated:
            self.episodes = torch.where(terminated, self.episodes + 1, self.episodes)
            self.keys = torch.where(
                terminated, rng.episode_key_torch(self.seeds, self.episodes), self.keys)
            self.t = torch.where(terminated, torch.zeros_like(self.t), self.t)
            self._reset_instances(terminated)
            if self.debug:
                self._check_invariants()
        return self.observe(), rewards, terminated, snapshot

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance all instances one step.

        Returns ``(obs, rewards, terminated, info)``. For instances where
        ``terminated[i]`` is True, ``obs[i]`` is the first observation of the
        next episode (auto-reset); the terminal state is in
        ``info["final_state_json"][i]`` (unless ``emit_final_states=False``).
        """
        obs, rewards, terminated, snapshot = self._core(actions)
        info: dict = {}
        if snapshot is not None and bool(terminated.any()):
            indices = torch.nonzero(terminated, as_tuple=False).flatten().tolist()
            info["final_state_json"] = {
                i: self.snapshot_slice_to_json(snapshot, i) for i in indices
            }
        return obs, rewards, terminated, info

    def snapshot_slice_to_json(self, snapshot: dict[str, torch.Tensor], i: int) -> dict:
        """Serialize instance ``i`` from a pre-reset state snapshot (the
        terminal state delivered in ``info["final_state_json"]``). Must
        produce the same schema-conformant shape as :meth:`slice_to_json` —
        if you override ``slice_to_json`` for nested state, override this
        consistently (it reads from ``snapshot`` because the live tensors
        already hold the next episode)."""
        return {name: tensor[i].tolist() for name, tensor in snapshot.items()}

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def _reset_instances(self, mask: torch.Tensor) -> None:
        """Initialize state tensors for instances where ``mask`` (Bool[N]) is
        True, leaving others bit-untouched. Called by ``reset`` (all True)
        and by auto-reset (terminated mask). ``self.keys``/``self.t`` are
        already updated; initialization draws use step 0 with dedicated
        slots. Allocate full-size tensors in the first call (during
        ``reset``), then masked-fill with ``torch.where``. Must stay
        branch-free on instance data (compilable)."""

    @abstractmethod
    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the transition for ALL instances (masking internally as
        needed). Returns ``(rewards Float[N], terminated Bool[N])``. Must not
        touch ``self.t``/``self.keys``/``self.episodes``."""

    @abstractmethod
    def observe(self) -> torch.Tensor:
        """Observation tensor ``[N, ...]`` for the current state."""

    @abstractmethod
    def state_tensors(self) -> dict[str, torch.Tensor]:
        """The serializable state as {schema field name: Tensor[N, ...]}.
        Include the in-episode step counter (``{"t": self.t}``) if it is part
        of the schema. Used for terminal-state capture and the default
        ``slice_to_json``."""

    def slice_to_json(self, i: int) -> dict:
        """Extract instance ``i`` as a dict conforming to schema.json's
        ``state`` definition — the representation the differential test
        compares against the reference's ``to_json``. Default: one flat field
        per ``state_tensors`` entry; override for nested/structured states."""
        return {name: tensor[i].tolist()
                for name, tensor in self.state_tensors().items()}

    # -- invariants --------------------------------------------------------

    def _check_invariants(self) -> None:
        for name, fn in self._invariants:
            ok = fn(self)
            if bool(ok.all()):
                continue
            idx = int(torch.nonzero(~ok, as_tuple=False).flatten()[0])
            path = self._dump_failure(name, idx)
            raise InvariantViolation(name, idx, path)

    def _dump_failure(self, invariant_name: str, i: int) -> Path | None:
        try:
            self.failures_dir.mkdir(parents=True, exist_ok=True)
            path = self.failures_dir / (
                f"{type(self).__name__}_{invariant_name}_env{i}_{int(time.time() * 1000)}.json"
            )
            snapshot = {
                "invariant": invariant_name,
                "env_index": i,
                "seed": int(self.seeds[i]),
                "episode": int(self.episodes[i]),
                "t": int(self.t[i]),
                "state": self.slice_to_json(i),
            }
            path.write_text(json.dumps(snapshot, indent=2))
            return path
        except Exception:
            return None  # never let failure-dumping mask the real violation
