"""Episode picking: which trajectories are worth looking at.

All functions take trajectory FILE PATHS (not env objects — visualization
never touches simulators) and return paths ordered most-interesting-first.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from simulacrum.traj.reader import read_trajectory
from simulacrum.traj.writer import Trajectory


def _paths(source) -> list[Path]:
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        # Skip export-pack companions that live alongside trajectory files
        # (manifest.json, schema.json, *.flat.json are not trajectories and
        # would crash read_trajectory).
        return sorted(
            p for p in Path(source).glob("*.json*")
            if not p.name.startswith("manifest")
            and p.name != "schema.json"
            and not p.name.endswith(".flat.json"))
    return [Path(p) for p in source]


def worst_return(source, k: int = 5) -> list[Path]:
    """The k trajectories with the lowest total return."""
    scored = [(read_trajectory(p, validate=False).total_return, p) for p in _paths(source)]
    scored.sort(key=lambda pair: pair[0])
    return [p for _, p in scored[:k]]


def highest_td_error(source, td_errors: dict[str, float], k: int = 5) -> list[Path]:
    """The k trajectories with the highest supplied TD-error score.

    ``td_errors`` maps trajectory filename (or full path) to a scalar computed
    by the training side — the framework does not compute TD errors itself.
    """
    def score(p: Path) -> float:
        return td_errors.get(str(p), td_errors.get(p.name, float("-inf")))
    ranked = sorted(_paths(source), key=score, reverse=True)
    return [p for p in ranked[:k] if score(p) != float("-inf")]


def invariant_failures(failures_dir: str | Path) -> list[Path]:
    """Snapshot files dumped by invariant violations (see BatchedEnv), newest
    first. Each contains {invariant, env_index, seed, episode, t, state} —
    the ``state`` is schema-conformant and renderable directly."""
    d = Path(failures_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def random_sample(source, k: int = 5, seed: int = 0) -> list[Path]:
    paths = _paths(source)
    return random.Random(seed).sample(paths, min(k, len(paths)))


def split_episodes(traj: Trajectory) -> list[list[dict]]:
    """Split a (possibly multi-episode) trajectory's steps on termination
    boundaries. Each element is the list of step records of one episode; the
    terminal step (with its TERMINAL state) is the last entry."""
    episodes: list[list[dict]] = [[]]
    for step in traj.steps:
        episodes[-1].append(step)
        if step["terminated"]:
            episodes.append([])
    if episodes and not episodes[-1]:
        episodes.pop()
    return episodes


def load_failure_state(path: str | Path) -> dict:
    """The renderable state dict from an invariant-failure snapshot."""
    return json.loads(Path(path).read_text())["state"]
