# Troubleshooting

Symptom, likely cause, what to do.

---

## `test_differential` diverges at step 0

Before any transition ran, so this is a **reset** problem.

| Look at | Because |
|---|---|
| The field named in the diff | The diff is field-level; read it first |
| Reset slot numbers | The two implementations may be using different slots for the same draw |
| `index` on repeated draws | Missing `index=k` makes every item draw the same number; all your items land on one cell |
| The step used for reset draws | Reset draws use step 0 in **both** implementations |
| Order of `to_json` vs `slice_to_json` | Field names and dtypes must match `schema.json` exactly |

If the diff shows `energy: 0.9800000190734863 vs 0.9800000004470348`, that is
dtype drift, not a logic error. One side is computing in float64 where the
spec says float32.

---

## `test_differential` diverges a few steps in

A **transition** problem. In rough order of likelihood:

1. **The per-step draw is keyed on the wrong counter.** Pre-move `t` versus
   post-move `t + 1`. This is the most common failure by a wide margin, and it
   is nasty because the draws stay perfectly uniform. The environment looks
   completely healthy. Check that `spec.md` says which, and that both sides
   obey it.
2. **Operation order differs from the spec.** Clamping before a rotation
   instead of after, normalizing before scaling. If the spec does not pin the
   order down, fix the spec first, otherwise you are just choosing which
   implementation to bless.
3. **Float order of operations.** `float32(a) / float32(b)` is not always
   `float32(a / b)`.
4. **A `torch.where` branch that is not safe everywhere.** It evaluates both
   sides for every instance; an out-of-bounds gather on the not-taken branch
   corrupts the taken one.

The failure message names the first divergent step and dumps both
trajectories into `failures/differential/`. Render them side by side. The
step before the divergence usually tells you more than the divergence itself.

---

## Diverges only at an episode boundary

The auto-reset boundary. Remember the convention: the batched env's returned
observation for a terminated instance is the **first observation of its next
episode**, and the terminal state arrives separately in
`info["final_state_json"]`. The reference env does not auto-reset at all.

If you overrode `slice_to_json`, you must override `snapshot_slice_to_json`
consistently. Terminal states are read from a pre-reset snapshot, not from
the live tensors, which already hold the next episode.

---

## Passes at n=1, fails `test_batch_independence`

Cross-instance leakage. Some operation is mixing data between instances.

Search `fast.py` for:

- **`keepdim=True`**: the usual culprit. A `[N, 1]` meeting a `[N]`
  broadcasts to `[N, N]`, and a later reduction hides it.
- **A `[N]` mask used without `unsqueeze`** against `[N, D]` or `[N, K]` state.
- **Any reduction over the batch axis**: `.sum(0)`, `.max()` with no `dim`,
  `.mean()`, which makes one instance's result depend on its neighbours.
- **In-place writes through boolean-index views** that alias another tensor.

The failure names the first instance and step that disagreed with its solo
run. This class of bug is invisible at `N = 1` by construction, so do not
expect the differential test to help you here. That is exactly why this test
exists. See [the cookbook](vectorization-cookbook.md#the-silent-one).

---

## `test_invariant_sweep` dumps a snapshot

An invariant failed under a long random rollout. `failures/` now holds a JSON
snapshot with `{invariant, env_index, seed, episode, t, state}`.

```python
from simulacrum.traj import picker
state = picker.load_failure_state("myenv/failures/....json")
print(render_state_text(state))
```

The state is schema-conformant, so your normal renderer works on it. Reproduce
by resetting that seed and episode and stepping to `t`.

If the invariant itself is wrong rather than the environment, fix `spec.md`
first and then the code. An invariant that does not match the spec is a
worse bug than the one it was going to catch.

---

## `test_invariant_sweep` fails with "no invariants registered"

You wrote none. `spec.md` requires an enumerated invariant list, and each
entry should be an `@invariant` method on the batched env returning `Bool[N]`,
True where it **holds**.

---

## `test_auto_reset` skips

No episode terminated within the step budget, so the auto-reset path was
never exercised. Make episodes terminate under random actions (a step cap,
a resource that depletes, anything) or raise `n_steps`. A skip here means one
of the trickiest parts of the framework went unchecked.

---

## `test_scripted_policies` fails after a change you believe is correct

The expected return is a **regression gate**, and if you deliberately changed
the reward function it is supposed to fail. Re-measure and update the
constant, and say in a comment that the number is measured rather than
derived, so the next person knows what kind of claim it is.

If you did not change the reward function, believe the test.

---

## `test_throughput` fails `min_speedup`

The ratio divides by your reference implementation, so a fast reference
deflates it and a noisy machine moves it. forager measured 13× and 18× on the
same container minutes apart.

Check the absolute steps/s in the report before assuming a regression. If the
absolute number is fine, lower `min_speedup`. It is a smoke alarm for
accidentally-scalar code, not a benchmark.

If the absolute number really did drop: confirm `benchmark_factory` still
sets `compile=True`, and check whether something introduced a Python-level
branch on instance data into the step path.

---

## `require_fresh_report` says STALE

An environment source file has a newer mtime than the report. That is the
gate working. Re-run `simulacrum validate`.

Note that `-k` subset runs deliberately do not write a report, so a subset run
will not clear this.

---

## The report is missing after a run that looked fine

If the battery errors before the session-finish hook (a collection error, an
import failure) no report is written, and the CLI refuses to trust a stale
one left over from an earlier run. Read the pytest output above the error
message; the real failure is there.

---

## `torch.compile` warnings, or the compiled env behaves differently

`test_benchmark_factory_parity` bit-checks the compiled path against eager, so
a genuine difference will fail loudly. If it passes and you still see
warnings, they are usually graph breaks from Python-level control flow in the
step path; the core must stay data-independent.

`debug=True` always forces the eager path, so a bug that only appears
compiled will not show up in a debug run.
