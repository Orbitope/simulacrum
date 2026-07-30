# Trajectories and visualization

**Simulators never render.** They emit trajectory files; everything visual
consumes those files. This keeps rendering out of the hot loop, lets a
heavyweight renderer live outside Python entirely, and means a bug in your
renderer can never be a bug in your environment.

---

## Trajectory files

`TrajectoryWriter` produces JSON (or JSONL for long runs) containing
metadata, an initial state, and a step record per step:

```python
from simulacrum.traj.writer import TrajectoryWriter

writer = TrajectoryWriter(env_name, seed, ref.to_json(state), env_root=root)
for t in range(n_steps):
    state, reward, term, _ = ref.step(action)
    writer.record(t, episode, ref.to_json(state), action, reward, term)
writer.write("run.json")
```

Metadata records the env name, format version, simulacrum version, seed,
source (`reference` or `batched`), instance index and git SHA: enough to
reproduce the run later.

Reading validates against your schema:

```python
from simulacrum.traj.reader import read_trajectory

traj = read_trajectory("run.json", schema=schema)   # raises on schema violation
traj.total_return                                   # sum of rewards
```

The battery already writes trajectories for you when a differential test
diverges, into `failures/differential/`, one per side, ready to render.

---

## Picking episodes worth looking at

Rendering everything is useless. `simulacrum.traj.picker` takes **file paths**
and returns them most-interesting-first:

```python
from simulacrum.traj import picker

picker.worst_return("trajectories/", k=5)         # lowest total return
picker.highest_td_error("trajectories/", scores)  # your TD errors, mapped by filename
picker.invariant_failures("myenv/failures")       # snapshots from invariant violations
picker.random_sample("trajectories/", k=5)        # a control group
```

`highest_td_error` takes scores you compute on the training side. The
framework does not compute TD errors itself.

`invariant_failures` reads the snapshots `BatchedEnv` dumps when an invariant
trips under `debug=True`. Each contains `{invariant, env_index, seed, episode,
t, state}`, and the `state` is schema-conformant, so it renders with the same
code as any other frame:

```python
state = picker.load_failure_state(path)
print(render_state_text(state))
```

`split_episodes(traj)` splits a multi-episode trajectory on termination
boundaries, with the terminal state as the last entry of each episode.

---

## Rendering

Your environment supplies two functions in `render.py`:

```python
def render_state_text(state: dict) -> str: ...
def render_state_mpl(state: dict, ax) -> None: ...
```

Both take a **schema-conformant state dict**. The rules are absolute:
renderers contain zero game logic and never import `reference.py` or
`fast.py`. If your renderer needs to know something, it belongs in the state.

The generic playback layer then does the rest:

```python
from simulacrum.viz.terminal import render_trajectory_text
from simulacrum.viz.frames import render_frames, render_trajectory_video

render_trajectory_text(traj, render_state_text)
render_frames(traj, render_state_mpl, out_dir="frames/")
render_trajectory_video(traj, render_state_mpl, "episode.mp4", fps=8)
```

Video stitching needs `ffmpeg` on PATH; matplotlib is an optional dependency
(`pip install -e '.[viz]'`).

---

## External renderers: the export pack

Heavyweight renderers (Unity, a web viewer) live outside Python and consume
files only.

```bash
simulacrum export-pack myenv/schema.json trajectories/*.json -o pack/
```

produces:

```
pack/
  manifest.json            # index of everything, flat and typed
  schema.json              # your env's schemas
  <name>.trajectory.json   # canonical, schema-conformant
  <name>.flat.json         # flattened parallel-array companion
```

### Why there are two representations

The canonical file carries arbitrarily nested state per your schema. Unity's
built-in `JsonUtility` cannot deserialize nested dictionaries or polymorphic
JSON, only flat `[Serializable]` objects with primitive fields and typed
lists. So the pack also emits a flat companion in which every **scalar** state
field becomes a parallel array over steps, alongside `rewards`, `actions`,
`terminal_step_indices`, and derived series (`reward`, `cumulative_return`)
shaped for direct feeding into a graph or HUD widget.

Non-scalar fields cannot be projected that way and are listed by name in
`skipped_fields`; a renderer that needs them must parse the canonical file
with a real JSON library. The flat file says so explicitly rather than
silently emitting zeros. Likewise `has_scalar_actions` is `false` when
actions are structured.

In the flat form, `values[0]` is the initial state and `values[i+1]` is the
post-step state of step `i`.

Nothing in the pack is Unity-specific. Any external renderer can consume
either representation.
