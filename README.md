# simulacrum

A framework for developing reinforcement-learning environments as batched
tensor simulations, with correctness guaranteed by differential testing
against slow, readable reference implementations.

**[→ Read the interactive article](https://orbitope.github.io/simulacrum/)**:
how to vectorize an environment without giving up the one you can read, with
everything running live in your browser.

## The idea

In most RL training loops, collecting experience is the bottleneck. Rewriting
the simulator so every instance advances in one tensor operation removes that
ceiling. On the examples here, `toywalk` goes from 385,000 steps/s to 35.3
million, a 92× speedup.

The problem is that the fast version is a *different program*: every `if`
becomes arithmetic that computes both outcomes and discards one. Nothing
checks that it is still the same environment, and a subtly wrong simulator
still produces a training curve that goes down. So the usual choice is to keep
the version you can reason about and train slowly, or take the speed and stop
being able to explain your results.

Simulacrum removes the choice. You write the environment **twice**: a
readable single-instance `reference.py` and a batched `fast.py`, both from a
single `spec.md`, and **never from each other**. A validation battery runs them
side by side and demands bit-identical behaviour. If they agree, either you
implemented the spec correctly twice, or you made the same misreading twice in
two very different programming styles.

The readable implementation stays the thing you debug against and explain
results with; the fast one is provably the same environment. Nothing trains
against an environment that has not passed.

## Install

```bash
pip install -e .
```

Python 3.10+, PyTorch 2.2+. `pip install -e '.[viz]'` adds matplotlib for
figure rendering.

## Use

```bash
simulacrum new myenv                      # scaffold an environment package
simulacrum validate myenv                 # run the full validation battery
simulacrum export-pack myenv/schema.json trajectories/*.json -o pack/
```

```python
from simulacrum.harness import require_fresh_report

require_fresh_report("myenv", strict=True)   # at the top of training scripts
```

## Documentation

| | |
|---|---|
| [Interactive article](https://orbitope.github.io/simulacrum/) | Start here. What the framework is for, interactively. |
| [User guide](docs/guide/) | Quickstart, walkthrough, cookbook, validation, troubleshooting. |
| [Architecture](docs/architecture.md) | The framework's own contracts: RNG, auto-reset, trajectories. |

## Examples

No environment ships with the framework; yours lives in its own repository.
Two examples come with the source, and both pass the full battery.

- **[`examples/toywalk`](examples/toywalk)**: deliberately boring. A 1-D walk
  with two scalar state fields; the smallest complete thing the framework
  accepts.
- **[`examples/forager`](examples/forager)**: deliberately shaped. An 8×8 grid
  with `[K, 2]` berry positions, per-item random draws and collection through a
  masked scatter, demonstrating the idioms a scalar toy cannot show, including
  the broadcasting mistakes `test_batch_independence` exists to catch.

## What the battery checks

Ten tests. Six of them must pass for an environment to be training-eligible.

| test | what it proves |
|---|---|
| `spec_contract` | the spec and schema exist and are complete |
| `differential` | reference and batched agree bit-for-bit, across episode boundaries |
| `batch_independence` | no cross-instance leakage; catches bugs invisible at n=1 |
| `invariant_sweep` | your stated invariants hold under long random rollouts |
| `auto_reset` | the episode boundary is exactly right |
| `determinism` | same seeds and actions reproduce bit-identically |
| `replay` | trajectories round-trip and reproduce themselves |
| `scripted_policies` | rewards mean what you think they mean |
| `benchmark_factory_parity` | the compiled config you train with matches the validated one |
| `throughput` | records steps/s; optionally gates on a speedup floor |

`simulacrum validate` writes `validation_report.json` with per-test outcomes,
versions, git SHA and throughput. It is gitignored on purpose. It attests
that *this* checkout passed on *this* machine, and a committed one would ride
along into checkouts where that is no longer true.

## License

MIT.
