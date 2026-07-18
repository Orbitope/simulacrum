"""Trajectory files: the only artifact simulators emit for consumers.

Canonical JSON layout (schema.json's ``trajectory`` definition)::

    {
      "metadata": {
        "env": str, "format_version": "1", "simulacrum_version": str,
        "seed": int, "source": "reference" | "batched", "instance": int|null,
        "git_sha": str|null, "created_at": iso8601 str
      },
      "initial": {"episode": 0, "state": <state>},
      "steps": [
        {"t": int, "episode": int, "state": <state>, "action": <action>,
         "reward": float, "terminated": bool},
        ...
      ]
    }

``t`` is the global step index of the run; ``episode`` is the episode the
step belongs to (the episode that was live when the action was taken).
``state`` is the post-step state — for a terminating step, the TERMINAL
state, not the auto-reset state.

JSONL layout for long runs: line 1 ``{"metadata": ...}``, line 2
``{"initial": ...}``, then one step object per line.

Renderers and all other consumers read these files; they never import the
environment.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simulacrum

FORMAT_VERSION = "1"


def _git_sha(root: str | Path | None) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root or ".", capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


@dataclass
class Trajectory:
    metadata: dict
    initial: dict
    steps: list[dict] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return sum(s["reward"] for s in self.steps)

    def to_obj(self) -> dict:
        return {"metadata": self.metadata, "initial": self.initial, "steps": self.steps}


class TrajectoryWriter:
    """Accumulate steps for one instance and write JSON or JSONL.

    Works identically for either implementation: the reference env passes
    ``to_json(state)`` dicts; the batched env passes ``slice_to_json(i)``.
    """

    def __init__(self, env_name: str, seed: int, initial_state: dict, *,
                 source: str = "reference", instance: int | None = None,
                 env_root: str | Path | None = None):
        self.traj = Trajectory(
            metadata={
                "env": env_name,
                "format_version": FORMAT_VERSION,
                "simulacrum_version": simulacrum.__version__,
                "seed": int(seed),
                "source": source,
                "instance": instance,
                "git_sha": _git_sha(env_root),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            initial={"episode": 0, "state": initial_state},
        )

    def record(self, t: int, episode: int, state: dict, action, reward: float,
               terminated: bool) -> None:
        self.traj.steps.append({
            "t": int(t), "episode": int(episode), "state": state,
            "action": action, "reward": float(reward), "terminated": bool(terminated),
        })

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            with path.open("w") as f:
                f.write(json.dumps({"metadata": self.traj.metadata}) + "\n")
                f.write(json.dumps({"initial": self.traj.initial}) + "\n")
                for step in self.traj.steps:
                    f.write(json.dumps(step) + "\n")
        else:
            path.write_text(json.dumps(self.traj.to_obj(), indent=2))
        return path
