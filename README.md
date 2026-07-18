# simulacrum

A framework for developing reinforcement-learning environments as batched tensor
simulations, with correctness guaranteed by differential testing against slow,
readable reference implementations.

**Two implementations per environment, one spec.** Every environment provides a
single-instance reference implementation (plain Python, explicit control flow)
and a batched tensor implementation (PyTorch, `[N, ...]` leading dimension,
branching replaced by masking). Both are written from `spec.md` — never from
each other. Correctness means bit-identical behavior under the validation
battery.

```bash
pip install -e .
simulacrum new myenv          # scaffold a new environment package
simulacrum validate examples/toywalk   # run the full validation battery
```

See [docs/architecture.md](docs/architecture.md) for the philosophy, the RNG
contract, and the trajectory/rendering contracts. No environment ships with the
framework; `examples/toywalk` is a deliberately boring toy that exercises every
feature and serves as the template for real environments, which live in their
own repos.
