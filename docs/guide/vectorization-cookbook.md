# Vectorization cookbook

Translating scalar environment logic into batched tensor code, and the traps
that make it dangerous. Read this before writing `fast.py`.

All state lives in tensors with a leading batch dimension `[N, ...]`. There
is no per-instance Python object, and the step path must contain **no
data-dependent Python branching on instance state**.

---

## Branch → mask

Anywhere the reference implementation writes:

```python
if condition:
    x = a
else:
    x = b
```

the batched implementation writes:

```python
x = torch.where(condition, a, b)      # condition: Bool[N]
```

`torch.where` **evaluates both branches for every instance.** Two consequences
you must respect:

**Both branches must be safe to compute everywhere** — no out-of-bounds
indexing, no division by zero, no NaN-producing operation on the not-taken
side. Clamp or sanitize indices *before* gathering, then mask the result:

```python
# wrong: idx may be -1 for instances that aren't gathering
val = torch.where(has_target, table[idx], default)

# right: make the gather safe for everyone, then choose
safe = idx.clamp(min=0)
val = torch.where(has_target, table[safe], default)
```

**Discarded random draws are harmless.** The RNG is stateless, so drawing for
all N instances and masking cannot perturb anything. Always draw for the full
batch — never try to draw "only for the instances that need it".

---

## Broadcasting hygiene

This is where the real bugs live, and where the framework's most-warned-about
failure mode comes from.

A `[N]` mask against `[N, D]` state does **not** do what you want. NumPy and
PyTorch right-align shapes, so `(N,)` becomes `(1, N)` and you either get a
loud error or — worse — a silent `[N, N]`.

**Lift the mask to the rank of the state it selects:**

```python
m = mask.unsqueeze(-1)                  # [N, 1]
self.pos = torch.where(m, new_pos, self.pos)                  # [N, 2]
self.kinds = torch.where(m, new_kinds, self.kinds)            # [N, K]
self.berries = torch.where(m.unsqueeze(-1), new_b, self.berries)  # [N, K, 2]
```

Comparing a per-instance vector against per-item state needs the same care:

```python
# berries [N, K, 2] vs pos lifted to [N, 1, 2] -> [N, K]
on_cell = (self.berries == self.pos.unsqueeze(1)).all(-1)
```

### The silent one

Some shape mistakes error immediately, which is a gift. The dangerous ones
broadcast into an `[N, N]` matrix that a later reduction collapses back into
a plausible `[N]`. One keyword:

```python
# correct
alive_count = self.alive.sum(-1)                      # [N]
terminated  = (alive_count == 0) | (self.energy <= 0.0)
#              [N]                  [N]          -> [N]   ✓

# one keyword added
alive_count = self.alive.sum(-1, keepdim=True)        # [N, 1]
terminated  = (alive_count == 0) | (self.energy <= 0.0)
#              [N, 1]               [N]          -> [N, N] ✗
```

Instance *i* now terminates as soon as **any** instance in the batch runs out
of energy.

Here is why this matters more than an ordinary bug: at `N = 1` the accidental
matrix is 1×1, which is exactly the right answer. The differential test runs
at `N = 1`, because the reference implementation only knows how to be one
instance. **The bug is not merely missed there; it is mathematically
invisible.** `test_batch_independence` exists for precisely this class, and
it is the reason it compares a solo run against an in-batch run instead of
comparing against the reference at all.

Habits that help: put the expected shape in a comment on any line that
combines differently-ranked tensors, and be suspicious of every `keepdim=True`.

---

## Dtype discipline

Match `schema.json` exactly. Dtype drift breaks bit-identity, and the
differential test reports it as a dtype mismatch — which is a much better
error than a value that is merely close.

```python
# spec: divisions computed in float32
self.pos[:, 0].to(torch.float32) / float(G - 1)      # float32 ✓
self.pos[:, 0] / (G - 1)                             # promotes -> float32? int? ✗
```

Common sources: an integer intermediate promoting to int32 on some backends;
a Python float constant pulling a float32 tensor to float64; `sum()` on a
bool tensor producing int64 where you wanted the count as float32.

Where the spec says float32, make the narrowing explicit at every step rather
than trusting promotion rules.

---

## Masked scatter

Updating per-item state for only some instances, without touching the rest:

```python
collected = on_cell & self.alive              # [N, K]
self.alive = self.alive & ~collected          # whole-tensor, no indexing
```

Prefer whole-tensor boolean algebra over `state[mask] = value[mask]`. Boolean
indexing works, but it produces views, and in-place operations on a view that
aliases another tensor will corrupt state in ways that are miserable to
debug. If you must use it, `clone()` first.

For a reduction over items that must match a scalar accumulation, multiply by
the mask rather than filtering — masked-out entries contribute an exact `0.0`,
so the sums agree:

```python
gained = (self._gains[self.kinds] * collected.to(torch.float64)).sum(-1)
```

---

## The auto-reset contract

`BatchedEnv.step` runs a branch-free core:

1. `_step_impl(actions)` → `(rewards, terminated)`, mutating state for all
   instances; then the base class increments `self.t`.
2. The post-transition state is snapshotted — this is where terminal states
   are captured.
3. Terminated instances get a new episode key, `self.t` zeroed, and
   `_reset_instances(terminated)` called.
4. `observe()` runs on the **post-reset** state.

So the observation returned for a terminated instance is the **first
observation of its next episode**, and the terminal state is delivered
separately in `info["final_state_json"]`. This trips people up; the reference
env does not auto-reset at all, and the harness bridges the two conventions.

Your obligations:

- `_step_impl` must not touch `self.t`, `self.keys` or `self.episodes`.
- Use `self.t + 1` for a post-move step-cap check.
- `_reset_instances` must leave non-terminated neighbours **bit-untouched**.
  `test_auto_reset` checks exactly that.

---

## Performance

Once it is correct:

```python
BatchedEnv(n, compile=True, emit_final_states=False)
```

- **`compile=True`** wraps the tensor core in `torch.compile`. Eager
  small-op dispatch dominates batched stepping; fusing the core is worth
  5–10× on CPU. It is ignored when `debug=True`, and the battery bit-checks
  the compiled path against eager so the thing you train with is the thing
  that was validated.
- **`emit_final_states=False`** stops building terminal-state JSON dicts every
  step. At large N terminations happen almost every step and this costs real
  time; training that bootstraps from tensors does not need them.

Register that configuration as `benchmark_factory` in your `HarnessConfig` so
it is validated rather than assumed.

Two things to know about the numbers:

- `torch.compile` has a per-call overhead of roughly 25µs, which is the floor
  at moderate N. Throughput keeps scaling with N well past where it looks
  flat.
- The **speedup ratio** divides by your reference implementation, so a fast
  reference deflates it. toywalk reports 92× and forager 13× on the same
  machine — not because forager is badly vectorized, but because its
  reference is slower per step. Judge absolute steps/s too; the report records
  both.
