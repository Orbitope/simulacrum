"""Matplotlib frame rendering + ffmpeg stitching for trajectory files.

Requires the ``viz`` extra (matplotlib) and an ``ffmpeg`` binary on PATH for
video assembly. The env package supplies only ``render_state_mpl(state, ax)``;
everything else here is generic plumbing.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from simulacrum.traj.writer import Trajectory


def render_frames(traj: Trajectory,
                  render_state_mpl: Callable[[dict, "object"], None],
                  out_dir: str | Path,
                  figsize: tuple[float, float] = (6, 4),
                  dpi: int = 100) -> list[Path]:
    """Render every state (initial + each step) to numbered PNG frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = [(None, traj.initial["state"])] + [(s, s["state"]) for s in traj.steps]
    paths = []
    for i, (step, state) in enumerate(states):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        render_state_mpl(state, ax)
        if step is not None:
            ax.set_title(f"t={step['t']} ep={step['episode']} "
                         f"a={step['action']} r={step['reward']:+.2f}"
                         + ("  TERMINAL" if step["terminated"] else ""))
        else:
            ax.set_title("initial state")
        path = out_dir / f"frame_{i:06d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def stitch_video(frames_dir: str | Path, out_path: str | Path, fps: int = 8) -> Path:
    """Assemble frame_%06d.png files into a video via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — install it, or use the PNG frames in "
            f"{frames_dir} directly")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(Path(frames_dir) / "frame_%06d.png"),
         "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         str(out_path)],
        check=True)
    return out_path


def render_trajectory_video(traj: Trajectory,
                            render_state_mpl: Callable[[dict, "object"], None],
                            out_path: str | Path,
                            fps: int = 8,
                            figsize: tuple[float, float] = (6, 4)) -> Path:
    """Frames + stitch in one call (frames go to a temp dir)."""
    with tempfile.TemporaryDirectory() as tmp:
        render_frames(traj, render_state_mpl, tmp, figsize=figsize)
        return stitch_video(tmp, out_path, fps=fps)
