# forager — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

Where `toywalk` is deliberately minimal, forager is deliberately *shaped*: it
carries vector state (`[K, 2]` berry positions, `[K]` flags), draws the same
slot once per berry via the RNG `index` word, and collects berries through a
masked scatter. Those are the idioms a real environment needs and a scalar
env cannot demonstrate.

Constants: `G = 8` (grid is `G × G`, coordinates `0..G-1`), `K = 6` (berries),
`MAX_STEPS = 80`, `GUST_P = 0.15`, `START_ENERGY = 1.0`, `STEP_COST = 0.02`,
`GAINS = [0.1, 0.15, 0.3]` (indexed by berry kind), `N_KINDS = 3`,
`REWARD_SCALE = 10.0`, `REWARD_STEP = -0.1`.

`START_ENERGY`, `STEP_COST` and `GAINS` are float32 constants; `energy`
arithmetic is performed **in float32** (see #Energy). Rewards are float64.

## State space

| field     | dtype        | bounds            | meaning                                  |
|-----------|--------------|-------------------|------------------------------------------|
| pos       | int64[2]     | `[0, G-1]` each   | agent cell, `(x, y)`                     |
| berries   | int64[K, 2]  | `[0, G-1]` each   | berry cells, fixed for the episode       |
| kinds     | int64[K]     | `[0, N_KINDS-1]`  | berry kind; indexes `GAINS`              |
| alive     | bool[K]      | —                 | berry not yet collected                  |
| energy    | float32      | `<= ENERGY_MAX`   | remaining energy (may go negative)       |
| t         | int64        | `[0, MAX_STEPS]`  | in-episode step counter (RNG key)        |

`ENERGY_MAX = START_ENERGY + K * max(GAINS)`. Energy is only bounded above;
it goes negative on the step that exhausts it, and that state is terminal.

Serialized form: `schema.json` `$defs/state`. `pos` and `kinds` serialize as
flat integer arrays, `berries` as an array of `[x, y]` pairs, `alive` as an
array of booleans, `energy` as a number, `t` as an integer.

## Actions

Integer in `{0, 1, 2, 3}` — a compass direction, giving the intended delta:

| action | direction | delta `(dx, dy)` |
|--------|-----------|------------------|
| 0      | north     | `(0, +1)`        |
| 1      | east      | `(+1, 0)`        |
| 2      | south     | `(0, -1)`        |
| 3      | west      | `(-1, 0)`        |

## Observations

`float32[5]`:

```
[ pos_x / (G - 1),
  pos_y / (G - 1),
  energy,
  alive_count / K,
  t / MAX_STEPS ]
```

`alive_count` is the number of `True` entries in `alive` (an exact integer
sum). Every division is computed **in float32**: cast the integer to float32,
then divide by the float32 constant. `energy` is already float32 and is passed
through unchanged. This ordering is normative — it is what makes the two
implementations bit-identical.

## Rewards

Let `gained` be the total gain collected on this step (see #Collection), as a
float64 sum. Then:

```
reward = REWARD_SCALE * gained + REWARD_STEP
```

computed in float64, in exactly that order. A step that collects nothing
scores `REWARD_STEP` (−0.1).

## Termination

The episode terminates when, after the move and collection:

- every berry is collected (`alive_count == 0`), **or**
- `energy <= 0.0`, **or**
- the post-move step counter equals `MAX_STEPS`.

Energy alone guarantees termination: `START_ENERGY / STEP_COST = 50` steps, so
every episode ends within `MAX_STEPS` under any policy, including random ones.

## Reset

Drawn at step 0, each with its own slot (see #RNG slots):

- `pos = (randint(G) @ AGENT_X, randint(G) @ AGENT_Y)`, both at `index = 0`.
- For each berry `k` in `0..K-1`, drawn at `index = k`:
  `berries[k] = (randint(G) @ BERRY_X, randint(G) @ BERRY_Y)`,
  `kinds[k] = randint(N_KINDS) @ BERRY_KIND`.
- `alive[k] = True` for all `k`.
- `energy = START_ENERGY`, `t = 0`.

**Berries may overlap** each other and may share the agent's starting cell.
This is deliberate: it removes rejection sampling from the spec, and a step
that lands on a shared cell collects every live berry there at once — which is
exactly the multi-collection case the reduction in #Energy has to get right.

A berry sharing the agent's *starting* cell is not collected at reset;
collection happens only on a step (see #Collection).

## Collection

After the agent's post-move cell is known, berry `k` is collected on this step
iff `alive[k]` **and** `berries[k] == pos` (both coordinates equal). For every
collected berry, `alive[k]` becomes `False`.

`gained` is the sum of `GAINS[kinds[k]]` over the collected berries, **summed
in ascending `k` order**. The order is normative: it is what lets a scalar
accumulation and a vectorized reduction agree.

## Energy

```
energy <- energy - STEP_COST + gained
```

computed **in float32**, left to right, in exactly that order: subtract the
step cost first, then add the step's gain. `gained` is cast to float32 before
the addition.

## Invariants

1. `pos_in_bounds`: `0 <= pos[i] <= G - 1` for both coordinates, in every state.
2. `berries_in_bounds`: `0 <= berries[k][i] <= G - 1` for every berry and
   coordinate, in every state.
3. `kinds_valid`: `0 <= kinds[k] <= N_KINDS - 1` for every berry.
4. `step_counter`: `0 <= t <= MAX_STEPS` in every state.
5. `energy_bounded_above`: `energy <= ENERGY_MAX` in every state.
6. `fresh_episode`: `t == 0` implies every `alive[k]` is `True` and
   `energy == START_ENERGY`.

## RNG slots

| slot | name       | used at        | index   | distribution           |
|------|------------|----------------|---------|------------------------|
| 0    | AGENT_X    | reset (step 0) | 0       | `randint(G)`           |
| 1    | AGENT_Y    | reset (step 0) | 0       | `randint(G)`           |
| 2    | BERRY_X    | reset (step 0) | `k`     | `randint(G)`           |
| 3    | BERRY_Y    | reset (step 0) | `k`     | `randint(G)`           |
| 4    | BERRY_KIND | reset (step 0) | `k`     | `randint(N_KINDS)`     |
| 5    | GUST       | every step `t` | 0       | `bernoulli(GUST_P)`    |

Reset-time draws use step 0 with their own slots; per-step draws use the
current in-episode step `t`. Same slot + same step + same index = same draw,
in both implementations — that is the whole differential-testing contract.

`BERRY_X`, `BERRY_Y` and `BERRY_KIND` are drawn once **per berry**, using the
`index` word to separate the `K` draws of one slot at one step. Drawing all
`K` berries from one slot at `index = 0` would place every berry on the same
cell; the batched implementation must vary `index` across the berry axis, not
the batch axis.

## The transition, precisely

1. The intended delta is looked up from the action (see #Actions).
2. The `GUST` draw is keyed on the **pre-move** step counter `t`. If it is
   True, the delta is rotated 90° clockwise: `(dx, dy) -> (dy, -dx)`.
   (North→east, east→south, south→west, west→north.)
3. `pos <- clamp(pos + delta, 0, G - 1)`, per coordinate. The clamp is applied
   **after** the rotation, never before.
4. `t <- t + 1`.
5. Collection is resolved on the post-move cell (see #Collection), yielding
   `gained` and the updated `alive`.
6. `energy` is updated (see #Energy).
7. Reward and termination are evaluated on the post-move state.
