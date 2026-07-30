"""Rendering hooks for forager — consumed by simulacrum.viz.

Renderers contain ZERO game logic: they map a schema-conformant state JSON to
text or a matplotlib figure. They never import reference.py or fast.py.
"""

from __future__ import annotations

from forager import G, K, MAX_STEPS

# Berry kinds 0/1/2, lowest gain to highest.
_KIND_GLYPH = (".", "o", "O")
_KIND_COLOR = ("tab:green", "tab:orange", "tab:purple")


def render_state_text(state: dict) -> str:
    grid = [["·" for _ in range(G)] for _ in range(G)]
    for cell, kind, alive in zip(state["berries"], state["kinds"], state["alive"]):
        if alive:
            grid[cell[1]][cell[0]] = _KIND_GLYPH[kind]
    grid[state["pos"][1]][state["pos"][0]] = "A"

    # Row 0 is the bottom of the grid, so print rows in descending y.
    rows = ["|" + " ".join(grid[y]) + "|" for y in range(G - 1, -1, -1)]
    left = sum(1 for a in state["alive"] if a)
    rows.append(f"pos={tuple(state['pos'])} energy={state['energy']:+.3f} "
                f"berries={left}/{K} t={state['t']}")
    return "\n".join(rows)


def render_state_mpl(state: dict, ax) -> None:
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(-0.5, G - 0.5)
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.grid(True, color="lightgray", linewidth=0.5, zorder=0)
    ax.set_aspect("equal")

    for cell, kind, alive in zip(state["berries"], state["kinds"], state["alive"]):
        if not alive:
            continue
        ax.scatter([cell[0]], [cell[1]], s=140, marker="o",
                   color=_KIND_COLOR[kind], zorder=1)
    ax.scatter([state["pos"][0]], [state["pos"][1]], s=220, marker="s",
               color="tab:blue", zorder=2, label="agent")

    left = sum(1 for a in state["alive"] if a)
    ax.set_title(f"t = {state['t']} / {MAX_STEPS}   "
                 f"energy = {state['energy']:+.3f}   berries left = {left}/{K}",
                 fontsize=9)
