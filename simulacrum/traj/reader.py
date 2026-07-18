"""Schema-validating trajectory reader."""

from __future__ import annotations

import json
from pathlib import Path

from simulacrum import spec as spec_mod
from simulacrum.traj.writer import Trajectory


def read_trajectory(path: str | Path, schema: dict | None = None,
                    validate: bool = True) -> Trajectory:
    """Read a .json or .jsonl trajectory. If ``schema`` (the env's schema.json
    document) is provided and ``validate`` is True, the trajectory is
    validated against its ``trajectory`` definition on load."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as f:
            lines = [json.loads(line) for line in f if line.strip()]
        obj = {"metadata": lines[0]["metadata"], "initial": lines[1]["initial"],
               "steps": lines[2:]}
    else:
        obj = json.loads(path.read_text())
    if schema is not None and validate:
        spec_mod.trajectory_validator(schema).validate(obj)
    return Trajectory(metadata=obj["metadata"], initial=obj["initial"], steps=obj["steps"])
