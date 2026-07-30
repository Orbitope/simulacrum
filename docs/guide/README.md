# simulacrum user guide

How to build your own environment with simulacrum, and how to keep it honest.

New here? The [interactive article](https://orbitope.github.io/simulacrum/)
explains what this framework is for and why it works, with the environment
running live in your browser. This guide is the dry version.

## Which document do I want

| I want to… | Read |
|---|---|
| get something running in ten minutes | [quickstart.md](quickstart.md) |
| build a real environment, start to finish | [writing-an-environment.md](writing-an-environment.md) |
| work out what my random draws should look like | [rng-and-slots.md](rng-and-slots.md) |
| translate scalar logic into batched tensor code | [vectorization-cookbook.md](vectorization-cookbook.md) |
| understand what the battery proves, and wire it into CI | [validation.md](validation.md) |
| look at trajectories, or hand them to Unity | [visualization.md](visualization.md) |
| fix a failing test | [troubleshooting.md](troubleshooting.md) |
| understand the framework's own contracts | [../architecture.md](../architecture.md) |

## The one-paragraph version

You write every environment **twice**: a slow, obviously-correct
single-instance version (`reference.py`), and a fast batched tensor version
(`fast.py`). Both are written from a single `spec.md`, and **never from each
other**. A validation battery then runs them side by side and demands
bit-identical behaviour. If they agree, either you implemented the spec
correctly twice, or you made the same misreading twice in two very different
programming styles — which is a much weaker coincidence than getting it right
once by luck. Nothing trains against an environment that has not passed.

## The two examples

Both live in [`examples/`](../../examples) and both pass the full battery.

**`toywalk`** is deliberately boring: a 1-D walk with two scalar state fields.
Read it first — it is the smallest complete thing the framework accepts, and
you can hold all of it in your head at once.

**`forager`** is deliberately *shaped*: an 8×8 grid with `[K, 2]` berry
positions, `[K]` flags, per-berry random draws and collection through a masked
scatter. Read it second — it demonstrates the idioms a real environment needs
and a scalar toy cannot show you, including the broadcasting mistakes that
`test_batch_independence` exists to catch.

Neither ships as part of the framework. `simulacrum` deliberately contains no
environments; yours lives in its own repository.
