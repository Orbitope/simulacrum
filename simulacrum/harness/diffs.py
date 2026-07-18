"""Field-level diff of two schema-conformant state JSON objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FieldDiff:
    path: str
    ref: Any
    fast: Any
    reason: str

    def __str__(self) -> str:
        return f"  {self.path or '<root>'}: reference={self.ref!r} fast={self.fast!r} ({self.reason})"


def _atol_for(path: str, atol: dict[str, float]) -> float:
    # Callers pass the path pre-wildcarded (numeric segments -> '*', matching
    # the keys spec.collect_atol emits), so a plain lookup suffices.
    return atol.get(path, 0.0)


def diff_states(ref: Any, fast: Any, atol: dict[str, float] | None = None,
                path: str = "") -> list[FieldDiff]:
    """Recursively diff two JSON values. ``atol`` maps state-field paths (as
    produced by ``simulacrum.spec.collect_atol``) to absolute tolerances;
    everything else must be exactly equal (bitwise, for floats)."""
    atol = atol or {}
    diffs: list[FieldDiff] = []

    if isinstance(ref, dict) and isinstance(fast, dict):
        for key in sorted(set(ref) | set(fast)):
            sub = f"{path}/{key}" if path else key
            if key not in ref:
                diffs.append(FieldDiff(sub, "<missing>", fast[key], "extra field in fast"))
            elif key not in fast:
                diffs.append(FieldDiff(sub, ref[key], "<missing>", "missing field in fast"))
            else:
                diffs.extend(diff_states(ref[key], fast[key], atol, sub))
    elif isinstance(ref, list) and isinstance(fast, list):
        if len(ref) != len(fast):
            diffs.append(FieldDiff(path, f"len {len(ref)}", f"len {len(fast)}", "length mismatch"))
        else:
            for i, (r, f) in enumerate(zip(ref, fast)):
                diffs.extend(diff_states(r, f, atol, f"{path}/{i}" if path else str(i)))
    elif isinstance(ref, bool) or isinstance(fast, bool):
        if ref is not fast:
            diffs.append(FieldDiff(path, ref, fast, "bool mismatch"))
    elif isinstance(ref, float) != isinstance(fast, float) and (
            isinstance(ref, (int, float)) and isinstance(fast, (int, float))):
        # int on one side, float on the other: numerically equal values would
        # slip through the float branch below, hiding exactly the dtype drift
        # the framework exists to catch.
        diffs.append(FieldDiff(path, ref, fast, "numeric type mismatch (int vs float — dtype drift)"))
    elif isinstance(ref, float) or isinstance(fast, float):
        tol = _atol_for(_wildcard_indices(path), atol)
        if tol > 0.0:
            if not math.isclose(float(ref), float(fast), abs_tol=tol, rel_tol=0.0):
                diffs.append(FieldDiff(path, ref, fast, f"exceeds declared x-atol {tol}"))
        elif not (float(ref) == float(fast) or (math.isnan(float(ref)) and math.isnan(float(fast)))):
            diffs.append(FieldDiff(path, ref, fast, "float mismatch (no x-atol declared)"))
    else:
        if ref != fast:
            diffs.append(FieldDiff(path, ref, fast, "value mismatch"))
    return diffs


def _wildcard_indices(path: str) -> str:
    """Replace numeric path segments with '*' so array-item tolerances match."""
    return "/".join("*" if seg.isdigit() else seg for seg in path.split("/"))


def format_diffs(diffs: list[FieldDiff], limit: int = 20) -> str:
    lines = [str(d) for d in diffs[:limit]]
    if len(diffs) > limit:
        lines.append(f"  ... and {len(diffs) - limit} more")
    return "\n".join(lines)
