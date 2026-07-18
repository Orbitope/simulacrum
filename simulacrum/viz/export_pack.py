"""External-renderer export pack.

Heavyweight renderers (Unity, web, ...) live outside Python: they consume
trajectory FILES and never import the environment. ``export_pack`` packages
everything such a renderer needs into one folder:

    <out_dir>/
      manifest.json          # index of everything in the pack (flat, typed)
      schema.json            # the env's state/trajectory JSON Schema
      <name>.trajectory.json # canonical trajectory files (schema-conformant)
      <name>.flat.json       # flattened parallel-array companions

DESIGN NOTE (the contentkit/Unity handoff, and why the flat companion exists):
Unity's built-in ``JsonUtility`` cannot deserialize nested dictionaries or
polymorphic JSON — only flat ``[Serializable]`` objects with primitive fields
and typed lists (see ContentKit's ElevenLabs subtitle importer, its one JSON
ingestion precedent). The canonical trajectory format (arbitrary nested state
per schema.json) is therefore unreadable with JsonUtility. The flat companion
projects every SCALAR state field into a parallel array over steps, plus
reward/action/termination arrays and derived scalar series (cumulative
return) shaped for ContentKit's ``CKDataGraph.Push(series, values)`` and
``CKHUD`` fields. Non-scalar state fields are listed in ``skipped_fields``;
a renderer that needs them must parse the canonical file with a real JSON
library (e.g. Newtonsoft) instead.

The pack is generic: nothing in it is Unity-specific, and any external
renderer can consume either representation.
"""

from __future__ import annotations

import json
import shutil
from itertools import accumulate
from pathlib import Path

from simulacrum.traj.reader import read_trajectory

PACK_FORMAT_VERSION = "1"


def _flatten(traj_obj) -> dict:
    steps = traj_obj.steps
    field_names: list[str] = sorted(traj_obj.initial["state"].keys())
    columns: list[dict] = []
    skipped: list[str] = []
    for name in field_names:
        values = [traj_obj.initial["state"][name]] + [s["state"][name] for s in steps]
        if all(isinstance(v, bool) for v in values):
            # Encoded as 0.0/1.0 so every column has one homogeneous float
            # array; consumers branch on dtype.
            columns.append({"name": name, "dtype": "bool",
                            "values": [1.0 if v else 0.0 for v in values]})
        elif all(isinstance(v, (int, float)) for v in values):
            dtype = "int" if all(isinstance(v, int) for v in values) else "float"
            columns.append({"name": name, "dtype": dtype,
                            "values": [float(v) for v in values]})
        else:
            skipped.append(name)

    rewards = [float(s["reward"]) for s in steps]
    cumulative = list(accumulate(rewards))

    # Only scalar numeric actions can be projected losslessly; anything else
    # must be read from the canonical trajectory file, and the flat file says
    # so explicitly rather than silently emitting zeros.
    has_scalar_actions = all(
        isinstance(s["action"], (int, float)) and not isinstance(s["action"], bool)
        for s in steps)
    actions = [float(s["action"]) for s in steps] if has_scalar_actions else []

    return {
        "format_version": PACK_FORMAT_VERSION,
        "env": traj_obj.metadata["env"],
        "seed": traj_obj.metadata["seed"],
        "source": traj_obj.metadata["source"],
        "num_steps": len(steps),
        "note": "state_fields values[0] is the initial state; values[i+1] is the post-step state of step i",
        "state_fields": columns,
        "skipped_fields": skipped,
        "has_scalar_actions": has_scalar_actions,
        "actions": actions,
        "rewards": rewards,
        "terminal_step_indices": [i for i, s in enumerate(steps) if s["terminated"]],
        "series": [
            {"name": "reward", "values": rewards},
            {"name": "cumulative_return", "values": cumulative},
        ],
    }


def export_pack(traj_paths, schema_path: str | Path, out_dir: str | Path,
                validate: bool = True) -> Path:
    """Build an export pack from trajectory files. Returns the pack dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads(Path(schema_path).read_text())
    shutil.copy(schema_path, out_dir / "schema.json")

    entries = []
    env_name = None
    seen_stems: set[str] = set()
    for path in map(Path, traj_paths):
        traj = read_trajectory(path, schema=schema, validate=validate)
        env_name = traj.metadata["env"]
        stem = path.name.replace(".jsonl", "").replace(".json", "")
        stem = stem.removesuffix(".trajectory")  # re-exporting a pack file
        if stem in seen_stems:  # distinct inputs must not overwrite each other
            n = 2
            while f"{stem}_{n}" in seen_stems:
                n += 1
            stem = f"{stem}_{n}"
        seen_stems.add(stem)
        traj_file = f"{stem}.trajectory.json"
        flat_file = f"{stem}.flat.json"
        (out_dir / traj_file).write_text(json.dumps(traj.to_obj(), indent=2))
        (out_dir / flat_file).write_text(json.dumps(_flatten(traj), indent=2))
        entries.append({
            "id": stem,
            "trajectory_file": traj_file,
            "flat_file": flat_file,
            "num_steps": len(traj.steps),
            "total_return": float(traj.total_return),
            "seed": int(traj.metadata["seed"]),
            "source": traj.metadata["source"],
        })

    manifest = {
        "format_version": PACK_FORMAT_VERSION,
        "env": env_name or "",
        "schema_file": "schema.json",
        "trajectories": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
