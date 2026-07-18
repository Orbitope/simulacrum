"""The renderer contract.

Rendering is a consumer, not a participant: simulators never render, and
renderers contain ZERO game logic. A renderer maps one schema-conformant
state JSON dict to an output (text, a matplotlib Axes, an image, ...). It
never imports the environment's reference or fast modules — its only input
is trajectory/snapshot files.

Environment packages provide their mapping in ``render.py`` as two
module-level functions (these aliases ARE the contract; the framework's
plumbing calls them and nothing else):

- ``render_state_text(state: dict) -> str``
- ``render_state_mpl(state: dict, ax) -> None``

The framework provides the plumbing around them: episode picking
(``simulacrum.traj.picker``), terminal playback (``simulacrum.viz.terminal``),
frame/video assembly (``simulacrum.viz.frames``), and the external-renderer
export pack (``simulacrum.viz.export_pack``).
"""

from __future__ import annotations

from typing import Any, Callable

# One state dict -> printable line(s).
RenderStateText = Callable[[dict], str]

# Draw one state dict onto a matplotlib Axes (second arg).
RenderStateMpl = Callable[[dict, Any], None]
