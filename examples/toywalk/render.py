"""Rendering hooks for toywalk — consumed by simulacrum.viz.

Renderers contain ZERO game logic: they map a schema-conformant state JSON to
text or a matplotlib figure. They never import reference.py or fast.py.
"""

from __future__ import annotations

from toywalk import L, MAX_STEPS


def render_state_text(state: dict) -> str:
    cells = []
    for x in range(-L, L + 1):
        if x == state["position"]:
            cells.append("A")
        elif x == L:
            cells.append("G")
        else:
            cells.append(".")
    return f"|{''.join(cells)}|  pos={state['position']:+d} t={state['t']}"


def render_state_mpl(state: dict, ax) -> None:
    ax.hlines(0, -L, L, color="lightgray", linewidth=4, zorder=0)
    ax.scatter([L], [0], marker="*", s=300, color="goldenrod", zorder=1, label="goal")
    ax.scatter([state["position"]], [0], s=200, color="tab:blue", zorder=2, label="agent")
    ax.set_xlim(-L - 0.5, L + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks(range(-L, L + 1))
    ax.text(-L, 0.6, f"t = {state['t']} / {MAX_STEPS}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
