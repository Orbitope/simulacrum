# RNG and slots

This is the part of the framework that makes everything else possible, and
the part most likely to bite you if you design it carelessly.

## The contract

Every random decision in your environment is a pure function of four numbers:

```
bits = hash(episode_key, step, slot, index)
episode_key = hash(instance_seed, episode_counter)
```

- **`episode_key`**: derived once per episode from `(seed, episode)`.
- **`step`**: the in-episode step counter `t` at which the draw happens.
  Reset-time draws use step 0.
- **`slot`**: a small integer naming *which* decision this is.
- **`index`**: disambiguates repeats of the same decision at the same step.

There is no stream, no cursor, no generator object. Draw the same four
numbers and you get the same bits, forever, in any order, in any language.

## Why it has to be this way

The batched implementation draws for **all N instances** and throws most of
the results away, because `torch.where` evaluates both branches. With a
conventional generator, those discarded draws would advance a shared stream
and the two implementations would desynchronise on the first masked step.

Because there is no stream state, three things become free:

- **Masking is safe.** A discarded draw was never part of a sequence.
- **Auto-reset needs no coordination.** A terminated instance increments its
  episode counter, re-derives its key, and carries on, with nothing to tell
  the reference implementation.
- **Porting is mechanical.** JAX's RNG is already counter-based; the article's
  JavaScript port is 90 lines of `BigInt`.

## Designing your slots

Enumerate every random decision in `spec.md`, then mirror it as an `IntEnum`:

```markdown
| slot | name       | used at        | index | distribution        |
|------|------------|----------------|-------|---------------------|
| 0    | AGENT_X    | reset (step 0) | 0     | randint(G)          |
| 1    | AGENT_Y    | reset (step 0) | 0     | randint(G)          |
| 2    | BERRY_X    | reset (step 0) | k     | randint(G)          |
| 5    | GUST       | every step t   | 0     | bernoulli(GUST_P)   |
```

Three rules:

**One slot per decision.** Two different random decisions at the same step
must not share a slot, or they will return the same bits and be perfectly
correlated. If your agent draws a spawn position and a spawn orientation,
those are two slots, not one slot used twice.

**Reset draws get their own slots.** Both a reset draw and the first
transition happen at step 0. If they share a slot they collide. Give
reset-time decisions dedicated slots and never reuse them per-step.

**Slot numbers are permanent.** Changing a slot number changes every draw
downstream of it, which changes every trajectory your environment has ever
produced. Append new slots; do not renumber.

## `index`: several draws of one decision

When one decision repeats at a single step (six berries to place, forty
cards to shuffle), vary `index`, not `slot`:

```python
for k in range(K):
    x = rng.draw_randint(self.key, 0, Slots.BERRY_X, G, index=k)
```

and in the batched version, the index varies along the **item** axis, never
the batch axis:

```python
bx = torch.stack(
    [rng.draw_randint_torch(self.keys, 0, Slots.BERRY_X, G, index=k)
     for k in range(K)], dim=-1)              # [N, K]
```

`K` is a compile-time constant, so that loop unrolls under `torch.compile`
and costs nothing at runtime.

Forgetting `index` is a real and quiet bug: every berry draws the same number
and lands on the same cell. The article's break-it lab has it as a toggle,
because it produces a perfectly functional environment that is simply not the
one you specified.

## The draw functions

Scalar (reference implementations):

```python
rng.episode_key(seed, episode)                        -> int
rng.draw_bits(key, step, slot, index=0)               -> int in [0, 2**64)
rng.draw_uniform(key, step, slot, index=0)            -> float64 in [0, 1)
rng.draw_randint(key, step, slot, n, index=0)         -> int in [0, n)
rng.draw_bernoulli(key, step, slot, p, index=0)       -> bool
```

Batched (tensor implementations): same names with a `_torch` suffix, taking
`self.keys` and returning tensors:

```python
rng.draw_bernoulli_torch(self.keys, self.t, Slots.GUST, GUST_P)   # Bool[N]
```

The two backends are guaranteed bit-identical for equal
`(key, step, slot, index)`. That guarantee is enforced by
`tests/test_rng.py`, which cross-checks them over large random and structured
inputs, and it is the foundation every other guarantee in the framework rests
on.

## Which step to key on

Per-step draws use the **pre-move** step counter, the value of `t` before
the transition increments it. Both implementations must agree:

```python
# reference: state.t is still the pre-move value here
if rng.draw_bernoulli(self.key, state.t, Slots.GUST, GUST_P):

# batched: self.t is incremented by the base class AFTER _step_impl returns
gust = rng.draw_bernoulli_torch(self.keys, self.t, Slots.GUST, GUST_P)
```

Getting this off by one is the single most common differential failure, and
it is nasty precisely because the draws remain perfectly uniform. The
environment looks entirely healthy and is simply a different environment. Say
which counter you mean in `spec.md`.

## Never do this

```python
import random, numpy as np, torch
random.random()          # no
np.random.rand()         # no
torch.rand(n)            # no
```

Any of these makes bit-identity impossible and silently removes the only
reason to write the environment twice.

## Draw distributions you do not have

The framework ships uniform, randint and bernoulli. For anything else, build
it from `draw_uniform` **identically in both implementations** and specify the
construction in `spec.md`. For instance, a categorical draw as a cumulative
comparison over a fixed probability vector in a stated order. If you build it
two different ways, the two ways will disagree in the low bits and you will
be declaring a float tolerance to paper over what is really a spec gap.
