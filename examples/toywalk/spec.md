# toywalk — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

Deliberately boring: a 1D "reach the goal" walk whose only purpose is to
exercise every simulacrum feature (stochastic transitions, reset randomness,
termination by goal AND by step cap, float observations, invariants).

Constants: `L = 5`, `MAX_STEPS = 60`, `SLIP_P = 0.1`, `START_RANGE = 2`,
`GOAL_REWARD = 10.0`, `STEP_REWARD = -1.0`.

## State space

| field    | dtype | bounds           | meaning                          |
|----------|-------|------------------|----------------------------------|
| position | int64 | [-L, L]          | location on the line; goal = +L  |
| t        | int64 | [0, MAX_STEPS]   | in-episode step counter (RNG key)|

Serialized form: `schema.json` `$defs/state` — `{"position": int, "t": int}`.

## Actions

Integer in {0, 1}: `0` = move left (−1), `1` = move right (+1).

## Observations

`float32[2]`: `[position / L, t / MAX_STEPS]`, both divisions computed **in
float32** (cast the integer to float32, then divide by the float32 constant).
This ordering is normative — it is what makes the two implementations
bit-identical.

## Rewards

If the move lands on `position == L` (the goal): `GOAL_REWARD` (+10.0).
Otherwise: `STEP_REWARD` (−1.0). Exact float constants; no arithmetic that
could differ between implementations, so no `x-atol` is declared.

## Termination

The episode terminates when, after the move, `position == L` (goal) or the
post-move step counter equals `MAX_STEPS` (step cap). Random actions always
terminate within `MAX_STEPS` steps.

## Reset

`position` drawn uniformly from `[-START_RANGE, +START_RANGE]` (5 values) as
`randint(2 * START_RANGE + 1) - START_RANGE`, using slot `INIT_POSITION` at
step 0. `t = 0`.

## Invariants

1. `in_bounds`: `-L <= position <= L` in every state.
2. `step_counter`: `0 <= t <= MAX_STEPS` in every state.
3. `init_in_start_range`: `t == 0` implies `|position| <= START_RANGE`.

## RNG slots

| slot | name          | used at        | distribution                          |
|------|---------------|----------------|---------------------------------------|
| 0    | INIT_POSITION | reset (step 0) | `randint(2 * START_RANGE + 1)`        |
| 1    | SLIP          | every step `t` | `bernoulli(SLIP_P)` — negates the move|

The transition, precisely: intended delta is −1 (action 0) or +1 (action 1);
if the SLIP draw (keyed on the pre-move step counter `t`) is True, the delta
is negated; `position ← clamp(position + delta, -L, L)`; `t ← t + 1`; then
rewards and termination are evaluated on the post-move state.
Reset-time draws use step 0 with their own slot; per-step draws use the
current in-episode step `t`. Same slot + same step + same index = same draw,
in both implementations.
