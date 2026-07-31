# Validation

What the battery proves, what it does not, and how to make it a gate rather
than a habit.

```bash
simulacrum validate myenv          # everything, writes the report
simulacrum validate myenv -k diff  # a subset, deliberately writes nothing
```

---

## The ten tests

### `test_spec_contract`
`spec.md` exists with every required section and no leftover TODOs;
`schema.json` is a valid JSON Schema with `state`, `action` and `trajectory`
definitions.

**Proves:** you did not skip the spec. **Does not prove:** the spec is any
good.

### `test_differential`
The keystone. K seeds × T steps, reference against batched-at-n=1 under one
action stream. Compares serialized state, observations, rewards and
termination flags at every step, including the terminal state on episode
boundaries and the post-auto-reset state. Reports the first divergence with a
field-level diff and dumps both trajectories.

**Proves:** the two implementations agree, so either you implemented the spec
twice correctly or made the same misreading twice in two very different
styles. **Does not prove:** anything about `N > 1`; see below.

### `test_batch_independence`
Instance *i* run alone must be bit-identical to instance *i* run inside a
batch of N. No reference implementation is involved.

**Proves:** no cross-instance leakage. This is the only test that can catch a
broadcast bug producing `[N, N]`, because such bugs are exactly correct at
`N = 1`, which is the only batch size `test_differential` ever runs. See
[the cookbook](vectorization-cookbook.md#the-silent-one).

### `test_invariant_sweep`
Long random-action rollouts over a large batch with `debug=True`, every
registered invariant checked batch-wide after every step.

**Proves:** your stated invariants hold in states the differential test never
reached. **Does not prove:** invariants you did not write. It fails outright
if you registered none.

### `test_auto_reset`
At the first mid-batch termination: the reset instance exactly matches a
fresh reference reset of the *next* episode; the returned observation is the
first of that next episode; untouched neighbours stay bit-identical.

**Proves:** the trickiest boundary in the framework. **Skips**, loudly, if
no episode terminated, which means your env does not terminate under random
actions and you should fix that.

### `test_determinism`
Same seeds, same actions, twice. Bit-identical.

### `test_replay`
A dumped reference trajectory, replayed through a fresh reference env,
reproduces itself. Also round-trips the writer/reader and validates against
`schema.json`.

### `test_scripted_policies`
Author-supplied policies with known expected returns, run on the reference.

**Proves:** something bit-identity cannot: that the reward function means
what you think. Two implementations can agree perfectly on a reward that is
simply wrong. Skips if you supply none, which is a waste of the cheapest
semantic check available.

### `test_benchmark_factory_parity`
If you supply a separate training-shape factory (`compile=True`,
`emit_final_states=False`), it must be bit-identical to the plain batched env.

**Proves:** the differential guarantee transfers to the configuration you
actually train with. Without it you validated a config nobody runs.

### `test_throughput`
Steps/sec for the reference and for the batched env at several batch sizes,
debug on and off. Asserts only `min_speedup` (default 0 = record-only).

---

## Required vs skippable

These must **pass**, not skip, for the environment to be training-eligible:

```
test_spec_contract   test_differential   test_batch_independence
test_invariant_sweep test_determinism    test_replay
```

The rest may skip without failing the gate, but each one that skips is a
check you chose not to have.

---

## Declaring a float tolerance

Correctness means bit-identical. Where a float field genuinely cannot be,
a reward computed via a vectorized reduction whose summation order differs
from a scalar loop, declare it **per field** in `schema.json`:

```json
"energy": { "type": "number", "x-atol": 1e-6 }
```

Undeclared float drift is a failure. Declared tolerances that are actually
exercised are recorded in the report under `tolerance_fields_used`.

**Do not declare tolerances pre-emptively.** Write the spec so both sides
perform the same operations in the same order, run the battery, and add a
tolerance only if you observe drift. forager was expected to need one for its
float32 energy accumulation (a scalar `+=` loop on one side and a
`(gains * mask).sum(-1)` reduction on the other) and turned out not to. Its
report shows `tolerance_fields_used: {}`, and the schema says so in the
field description. A tolerance you did not need is a weakened test.

`reward_atol` and `obs_atol` on `HarnessConfig` do the same job for rewards
and observations, which are not part of the state schema.

---

## Reading the report

```
  test_differential               PASS      1.83s
  ...
  reference:      174,770 steps/s
  batched n=8192,debug=False    2,277,929 steps/s
  speedup at largest batch: 13x
  overall: PASS (eligible for training)
```

`validation_report.json` also records the git SHA, Python/torch/numpy
versions, per-test durations and outcomes, throughput at every benchmark
batch size, and which tolerances were exercised.

On the speedup number: it divides by your reference implementation, so a fast
reference deflates it. toywalk reports 92× and forager 13× on the same
machine. The absolute steps/s is the more honest figure, and the ratio moves
between runs; forager measured 13× and 18× on the same container minutes
apart. Set `min_speedup` well below what you observe.

### The report is not a build artifact

`validation_report.json` is **gitignored**, and should stay that way. It
attests that *this* checkout passed on *this* machine with *these* library
versions. A committed report would ride along into checkouts where it is no
longer true, which is precisely the failure the staleness check exists to
prevent. Regenerate it in CI and locally; never commit it.

---

## The gate

```python
from simulacrum.harness import require_fresh_report

require_fresh_report("path/to/myenv", strict=True)
```

Returns `True` when the gate is satisfied. Otherwise it warns loudly on
stderr, or raises with `strict=True`, when the report is missing, failing,
or older than the environment's source files:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! VALIDATION GATE: env source files modified after the validation
!! report was written. Report is STALE
!! Run: simulacrum validate path/to/myenv
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Use `strict=True` for anything whose results you intend to trust. The warning
form is for exploratory work, and is easy to scroll past, which is the point
of making the other one raise.

---

## CI

```yaml
- run: pip install -e .
- run: simulacrum validate envs/myenv
```

The command exits non-zero when the battery fails or when the report says
`overall_pass: false`, so no extra assertion is needed.

Two notes for CI specifically. Set `min_speedup` conservatively. Shared
runners are noisy and a throughput assertion is the one test that can fail
for reasons unrelated to your code. And expect the last two tests to dominate
wall time: `torch.compile` warmup is most of a forager run.
