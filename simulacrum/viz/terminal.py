"""Terminal playback of trajectory files."""

from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

from simulacrum.traj.writer import Trajectory


def render_trajectory_text(traj: Trajectory,
                           render_state_text: Callable[[dict], str],
                           stream: TextIO = sys.stdout,
                           fps: float | None = None) -> None:
    """Print every state of a trajectory using the env package's
    ``render_state_text``. ``fps`` throttles playback (None = no delay)."""
    meta = traj.metadata
    stream.write(f"== {meta['env']} seed={meta['seed']} source={meta['source']} ==\n")
    stream.write(f"[init ep0]  {render_state_text(traj.initial['state'])}\n")
    for step in traj.steps:
        flag = "  TERMINAL" if step["terminated"] else ""
        stream.write(
            f"[t={step['t']:>4} ep{step['episode']}] "
            f"{render_state_text(step['state'])}"
            f"  a={step['action']} r={step['reward']:+.2f}{flag}\n")
        if fps:
            time.sleep(1.0 / fps)
    stream.write(f"== total return: {traj.total_return:+.2f} over "
                 f"{len(traj.steps)} steps ==\n")
