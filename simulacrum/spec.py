"""Loading and checking the per-environment contract files: spec.md + schema.json.

Every environment package provides:

- ``spec.md`` — the single source of truth both implementations are written
  from. Required sections (checked by :func:`load_spec`): State space, Actions,
  Observations, Rewards, Termination, Invariants, RNG slots.
- ``schema.json`` — a JSON Schema document with ``$defs`` for ``state``,
  ``action``, and ``trajectory``. The ``state`` definition is the canonical
  serialized form compared by the differential test. Float fields that
  legitimately cannot be bit-identical between implementations (e.g. computed
  via a vectorized reduction) may declare ``"x-atol": <float>``; every use of
  a tolerance is recorded in the validation report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema

# Singular stems, matched as substrings of headings, so "## Reward",
# "## Rewards", and "## Reward function" all satisfy the contract.
REQUIRED_SPEC_SECTIONS = [
    "state space",
    "action",
    "observation",
    "reward",
    "termination",
    "invariant",
    "rng slot",
]

REQUIRED_SCHEMA_DEFS = ["state", "action", "trajectory"]


class SpecError(ValueError):
    pass


def load_spec(env_root: str | Path) -> dict[str, str]:
    """Parse spec.md into {section-heading-lowercased: body}; raise SpecError
    listing any missing required section."""
    path = Path(env_root) / "spec.md"
    if not path.exists():
        raise SpecError(f"missing spec.md in {env_root}")
    text = path.read_text()
    sections: dict[str, str] = {}
    prose: dict[str, str] = {}  # body excluding fenced code blocks
    current = None
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            if current is not None:
                sections[current] += line + "\n"
            continue
        m = None if in_fence else re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            current = m.group(1).strip().lower()
            sections[current] = ""
            prose[current] = ""
        elif current is not None:
            sections[current] += line + "\n"
            if not in_fence:
                prose[current] += line + "\n"
    missing = [s for s in REQUIRED_SPEC_SECTIONS
               if not any(s in heading for heading in sections)]
    if missing:
        raise SpecError(f"spec.md missing required sections: {missing}")
    # TODO gate: whole-word match, prose only — a `# TODO` inside an
    # illustrative code fence or the word "TODOs" in a sentence doesn't count.
    todos = [h for h, body in prose.items() if re.search(r"\bTODO\b", body)]
    if todos:
        raise SpecError(f"spec.md still has TODO markers in sections: {todos}")
    return sections


def load_schema(env_root: str | Path) -> dict:
    """Load and sanity-check schema.json; returns the schema document."""
    path = Path(env_root) / "schema.json"
    if not path.exists():
        raise SpecError(f"missing schema.json in {env_root}")
    schema = json.loads(path.read_text())
    defs = schema.get("$defs", {})
    missing = [d for d in REQUIRED_SCHEMA_DEFS if d not in defs]
    if missing:
        raise SpecError(f"schema.json missing $defs: {missing}")
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def state_validator(schema: dict) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": "#/$defs/state"})


def trajectory_validator(schema: dict) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": "#/$defs/trajectory"})


def collect_atol(schema: dict) -> dict[str, float]:
    """Walk the ``state`` definition collecting declared ``x-atol`` keywords.

    Returns {json-path: atol}, where json-path is a ``/``-joined path into the
    state object ("" segments for array items are written ``*``), e.g.
    ``"velocity"`` or ``"cards/*/value"``.
    """
    out: dict[str, float] = {}

    def walk(node: dict, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "x-atol" in node:
            out[path] = float(node["x-atol"])
        for key, sub in node.get("properties", {}).items():
            walk(sub, f"{path}/{key}" if path else key)
        if "items" in node and isinstance(node["items"], dict):
            walk(node["items"], f"{path}/*" if path else "*")

    walk(schema["$defs"]["state"], "")
    return out
