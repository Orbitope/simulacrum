# simulacrum architecture

simulacrum is a framework for developing RL environments as batched tensor
simulations whose correctness is *proven*, not assumed. This document explains
the contracts. The API reference lives in the module docstrings
(`simulacrum.rng`, `simulacrum.batched`, `simulacrum.harness.battery`,
`simulacrum.viz.export_pack` are the load-bearing ones).

## 1. Two implementations, one spec

Every environment package contains:

| file | role |
|------|------|
| `spec.md` | The single source of truth: state space (dtypes/bounds), action semantics, observation encoding, reward function, termination rules, an **enumerated invariant list**, and an **enumerated RNG-slot table**. |
| `schema.json` | JSON Schema with `$defs` for `state`, `action`, `trajectory`. The `state` definition is the canonical serialized form the differential test compares. |
| `reference.py` | Slow, readable, single-instance (`ReferenceEnv`). Dataclass state, explicit `if`s, every rule traceable to a spec line. |
| `fast.py` | Batched tensor implementation (`BatchedEnv`). `[N, ...]` leading dim, branching replaced by `torch.where` masking, in-tensor auto-reset. |

Both implementations are written **from the spec, never from each other**. If
they agree bit-for-bit across the validation battery, either the spec was
implemented correctly twice, or the same misreading happened twice in two
different programming styles — which is what makes the differential test
strong evidence rather than a tautology.

Correctness = bit-identical behavior. Float fields that legitimately cannot
be bit-identical (e.g. a reward computed via a vectorized reduction) must
declare `"x-atol": <tol>` on the field in `schema.json`; every tolerance
actually exercised is recorded in the validation report. Undeclared float
drift is a failure.

## 2. The RNG contract (the keystone)

The hard part of differential testing a stochastic env is making two
implementations draw identical randomness. simulacrum uses a **stateless
counter-based RNG** (splitmix64 sponge — see `simulacrum/rng.py`):

```
bits = hash(domain, slot, index, episode_key, step)     # canonical word order
episode_key = hash(domain_ep, instance_seed, episode_counter)
```

- Every random decision in the env is a named **slot**, enumerated in spec.md.
- Reset-time draws use step 0 with dedicated slots; per-step draws use the
  in-episode step counter `t` (which is therefore part of env state).
- There is **no stream state**. Consequences:
  - The batched impl draws for ALL N instances and masks; discarded draws
    cannot corrupt parity. This is what makes `torch.where` (which evaluates
    both branches) safe.
  - Auto-reset needs no RNG coordination: the episode counter increments and
    keys re-derive deterministically.
  - A JAX port is mechanical — JAX's RNG is already counter-based.
- Two backends from the same constants: plain-Python ints (reference) and
  torch.int64 with two's-complement wrapping + emulated logical shifts
  (batched). `tests/test_rng.py` cross-checks them bitwise; that test is the
  foundation of every other guarantee.

## 3. Auto-reset and the observation boundary

`BatchedEnv.step` runs a branch-free tensor core: transition → invariant
check (debug) → terminal-state snapshot → episode re-key → masked in-tensor
reset of terminated instances → observe. The returned observation for a
terminated instance is the **first observation of its next episode**; the
terminal state is delivered as JSON in `info["final_state_json"]` (disable
with `emit_final_states=False` for training-shape throughput). The reference
env never auto-resets — the harness bridges the two conventions and tests the
boundary explicitly.

Because the core is branch-free it can be `torch.compile`d
(`BatchedEnv(..., compile=True)`, ~5-10x on CPU at large N). The battery's
benchmark-factory parity test bit-checks the compiled path against eager, so
the thing you train with is the thing that was validated.

## 4. Validation is a gate

`simulacrum validate <env>` runs the battery (see
`simulacrum/harness/battery.py`): spec contract, differential (K seeds × T
steps, field-level diff on first divergence, trajectory dumps), batch
independence (solo run vs in-batch, catches broadcast leaks the differential
test at n=1 *cannot see*), invariant sweep (`debug=True`, large batch),
auto-reset correctness, determinism, replay, scripted-policy returns,
throughput. It writes `validation_report.json` (pass/fail per test, versions,
git SHA, throughput at several batch sizes, tolerances exercised).

Training scripts call `simulacrum.harness.require_fresh_report(env_root)`,
which warns loudly (or raises with `strict=True`) when the report is missing,
failing, or older than the env's source files. **An environment without a
fresh passing report is not eligible for training.**

## 5. Trajectories and rendering

Simulators never render. They emit trajectory files (JSON, or JSONL for long
runs — see `simulacrum/traj/writer.py` for the exact layout), validated
against `schema.json` on read. All visualization consumes those files:

- `simulacrum.traj.picker` — episodes worth looking at: worst return,
  supplied TD-error scores, invariant-failure snapshots (auto-collected from
  `failures/`), random sample.
- `simulacrum.viz.terminal` / `simulacrum.viz.frames` — generic playback;
  the env package supplies only `render_state_text(state)` and
  `render_state_mpl(state, ax)` in `render.py`. Renderers contain zero game
  logic and never import the env's implementations.

### External renderers (the contentkit/Unity handoff)

Heavyweight renderers live outside Python and consume files only.
`simulacrum export-pack <schema> <trajs...> -o <dir>` builds a folder:
`manifest.json` + `schema.json` + canonical trajectory files + **flat
parallel-array companions**. The flat form exists because Unity's
`JsonUtility` (the only JSON path in contentkit today, per its ElevenLabs
subtitle importer) cannot parse nested/dict JSON: scalar state fields become
typed columns, and derived per-step series (`reward`, `cumulative_return`)
are shaped for direct feeding into ContentKit's `CKDataGraph.Push(series,
values)` / `CKHUD` scalar fields. A future ContentKit importer reads
`manifest.json`, picks a trajectory, and drives its timeline from the flat
file — without ever importing the environment. Nothing in the pack is
Unity-specific; any external renderer can use either representation.

## 6. Performance notes (measured on the toywalk example, CPU)

- Eager small-op dispatch dominates batched stepping; `compile=True` fuses
  the core (toywalk: ~2.5M → ~20M steps/s at N=1024).
- `torch.compile` per-call overhead (~25µs) is the floor at moderate N;
  throughput keeps scaling with N (toywalk: ~30M steps/s at N=8192).
- Building `final_state_json` dicts every step costs real time at large N
  (terminations happen almost every step); training configs that bootstrap
  from tensors should pass `emit_final_states=False`.
- A trivial env's pure-Python reference can itself run at ~1µs/step, which
  deflates the speedup *ratio* — judge batched performance in absolute
  steps/s too. The report records both.
