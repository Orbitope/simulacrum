"""``simulacrum new <name>``: scaffold an environment package.

Layout produced (the same layout examples/toywalk uses)::

    <name>/
      __init__.py
      spec.md           # TODO-marked; the single source of truth
      schema.json       # state/action TODO; trajectory def prefilled
      reference.py      # ReferenceEnv stub
      fast.py           # BatchedEnv stub
      render.py         # render_state stub for the viz layer
      tests/
        conftest.py     # harness_config fixture (one-line plugin hook)
        test_battery.py # the entire battery, star-imported

Write spec.md FIRST. Then reference.py and fast.py from the spec — never from
each other.
"""

from __future__ import annotations

from pathlib import Path

_SPEC_MD = '''# __NAME__ — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

## State space

TODO: every state variable with dtype and bounds, e.g.

| field    | dtype | bounds        | meaning                    |
|----------|-------|---------------|----------------------------|
| position | int64 | [-L, L]       | ...                        |
| t        | int64 | [0, max_steps]| in-episode step counter    |

The serialized form of the state is defined in `schema.json` (`$defs/state`).
Include the in-episode step counter `t` in state: RNG draws are keyed on it.

## Actions

TODO: action encoding and semantics (e.g. 0 = left, 1 = right).

## Observations

TODO: exact observation encoding (shape, dtype, normalization). Must be
identical between implementations — including dtype.

## Rewards

TODO: reward function, exactly. Note which (if any) float computations cannot
be bit-identical between a scalar and a vectorized implementation; those state
fields must declare `x-atol` in schema.json.

## Termination

TODO: termination conditions. Episodes should terminate under random actions
(the auto-reset test needs to observe a termination).

## Reset

TODO: initial state distribution. List which RNG slots reset uses (step 0,
dedicated slots).

## Invariants

TODO: an explicit enumerated list of properties that must hold in EVERY state,
e.g.

1. `position` is within [-L, L].
2. `t <= max_steps`.
3. terminated implies (goal reached or t == max_steps).

Each becomes an `@invariant` on the batched env, checked batch-wide every step
when `debug=True`.

## RNG slots

TODO: enumerate every random decision as a slot (IntEnum in the package):

| slot | name | used at              | distribution |
|------|------|----------------------|--------------|
| 0    | INIT_POSITION | reset (step 0) | randint(...) |
| 1    | SLIP | every step           | bernoulli(p) |

Reset-time draws use step 0 with their own slots; per-step draws use the
current in-episode step `t`. Same slot + same step + same index = same draw,
in both implementations — that is the whole differential-testing contract.
'''

_SCHEMA_JSON = '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "__NAME__ state and trajectory schemas",
  "$defs": {
    "state": {
      "type": "object",
      "description": "TODO: one serialized environment state. Mirror spec.md's state-space table exactly. Float fields that cannot be bit-identical across implementations may declare \\"x-atol\\".",
      "properties": {
        "t": { "type": "integer", "minimum": 0 }
      },
      "required": ["t"],
      "additionalProperties": false
    },
    "action": {
      "description": "TODO: one serialized action.",
      "type": "integer"
    },
    "trajectory": {
      "type": "object",
      "properties": {
        "metadata": {
          "type": "object",
          "properties": {
            "env": { "type": "string" },
            "format_version": { "type": "string" },
            "simulacrum_version": { "type": "string" },
            "seed": { "type": "integer" },
            "source": { "enum": ["reference", "batched"] },
            "instance": { "type": ["integer", "null"] },
            "git_sha": { "type": ["string", "null"] },
            "created_at": { "type": "string" }
          },
          "required": ["env", "format_version", "seed", "source", "created_at"]
        },
        "initial": {
          "type": "object",
          "properties": {
            "episode": { "type": "integer" },
            "state": { "$ref": "#/$defs/state" }
          },
          "required": ["episode", "state"]
        },
        "steps": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "t": { "type": "integer" },
              "episode": { "type": "integer" },
              "state": { "$ref": "#/$defs/state" },
              "action": { "$ref": "#/$defs/action" },
              "reward": { "type": "number" },
              "terminated": { "type": "boolean" }
            },
            "required": ["t", "episode", "state", "action", "reward", "terminated"]
          }
        }
      },
      "required": ["metadata", "initial", "steps"]
    }
  }
}
'''

_INIT_PY = '''"""__NAME__: a simulacrum environment package."""

from enum import IntEnum


class Slots(IntEnum):
    """RNG slots — MUST mirror the table in spec.md."""
    # TODO: enumerate every random decision, e.g.:
    # INIT_POSITION = 0   # reset-time (step 0)
    # SLIP = 1            # per-step
'''

_REFERENCE_PY = '''"""Readable single-instance reference implementation of __NAME__.

Written from spec.md ONLY. Style: dataclass state, explicit ifs, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from __NAME__.Slots.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulacrum import ReferenceEnv, rng

from __NAME__ import Slots


@dataclass
class State:
    t: int  # in-episode step counter (RNG draws are keyed on it)
    # TODO: fields per spec.md's state-space table


class __CLASS__Reference(ReferenceEnv):
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # Reset-time draws: step 0, dedicated slots, e.g.
        #   pos = rng.draw_randint(self.key, 0, Slots.INIT_POSITION, n)
        raise NotImplementedError("TODO: build initial State per spec.md #Reset")

    def step(self, action) -> tuple[State, float, bool, dict]:
        # Per-step draws: rng.draw_*(self.key, self.state.t, Slots.X)
        raise NotImplementedError("TODO: transition per spec.md")

    def observe(self, state: State):
        raise NotImplementedError("TODO: observation encoding per spec.md")

    def to_json(self, state: State) -> dict:
        raise NotImplementedError("TODO: serialize per schema.json $defs/state")

    def from_json(self, obj: dict) -> State:
        raise NotImplementedError("TODO: inverse of to_json")
'''

_FAST_PY = '''"""Batched tensor implementation of __NAME__.

Written from spec.md ONLY — do not look at reference.py while writing this.
State: tensors with leading [N] dim. Branching -> torch.where masking. See
simulacrum.batched docstrings for the idioms and the auto-reset contract.
"""

from __future__ import annotations

import torch

from simulacrum import BatchedEnv, invariant, rng

from __NAME__ import Slots


class __CLASS__Batched(BatchedEnv):
    def _reset_instances(self, mask: torch.Tensor) -> None:
        # Allocate state tensors on first call; masked-fill afterwards, e.g.:
        #   pos0 = rng.draw_randint_torch(self.keys, 0, Slots.INIT_POSITION, n)
        #   self.pos = torch.where(mask, pos0, self.pos)
        raise NotImplementedError("TODO: initialize per spec.md #Reset")

    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Draw for ALL instances (stateless RNG makes discarded draws safe):
        #   slip = rng.draw_bernoulli_torch(self.keys, self.t, Slots.SLIP, p)
        raise NotImplementedError("TODO: transition per spec.md; return (rewards, terminated)")

    def observe(self) -> torch.Tensor:
        raise NotImplementedError("TODO: observation encoding per spec.md")

    def slice_to_json(self, i: int) -> dict:
        raise NotImplementedError("TODO: serialize instance i per schema.json $defs/state")

    # TODO: one @invariant per entry in spec.md #Invariants, e.g.:
    # @invariant("position_in_bounds")
    # def _inv_position(self) -> torch.Tensor:
    #     return (self.pos >= -L) & (self.pos <= L)
'''

_RENDER_PY = '''"""Rendering hooks for __NAME__ — consumed by simulacrum.viz.

Renderers contain ZERO game logic: they map a schema-conformant state JSON to
text or a matplotlib figure. They never import reference.py or fast.py.
"""

from __future__ import annotations


def render_state_text(state: dict) -> str:
    raise NotImplementedError("TODO: text rendering of one state dict")


def render_state_mpl(state: dict, ax) -> None:
    raise NotImplementedError("TODO: draw one state dict onto a matplotlib Axes")
'''

_CONFTEST_PY = '''import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig

    from __NAME__.fast import __CLASS__Batched
    from __NAME__.reference import __CLASS__Reference

    return HarnessConfig(
        name="__NAME__",
        root=ENV_ROOT,
        reference_factory=__CLASS__Reference,
        batched_factory=lambda n, debug=False: __CLASS__Batched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=2),  # TODO
        # scripted_policies=[...],  # strongly recommended (see ScriptedPolicy)
        # min_speedup=100.0,        # enforce the vectorization win
    )
'''

_TEST_BATTERY_PY = '''from simulacrum.harness.battery import *  # noqa: F401,F403
'''

_TEMPLATES = {
    "spec.md": _SPEC_MD,
    "schema.json": _SCHEMA_JSON,
    "__init__.py": _INIT_PY,
    "reference.py": _REFERENCE_PY,
    "fast.py": _FAST_PY,
    "render.py": _RENDER_PY,
    "tests/conftest.py": _CONFTEST_PY,
    "tests/test_battery.py": _TEST_BATTERY_PY,
}


def scaffold(name: str, dest: str | Path = ".") -> Path:
    if not name.isidentifier() or name != name.lower():
        raise ValueError(
            f"environment name must be a lowercase Python identifier "
            f"(it becomes the package name), got {name!r}")
    class_name = "".join(part.capitalize() for part in name.split("_"))
    root = Path(dest) / name
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    for rel, template in _TEMPLATES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace("__NAME__", name).replace("__CLASS__", class_name))
    return root
