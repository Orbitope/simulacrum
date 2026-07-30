# Writing an environment

The full walkthrough, in the order you should actually do it. Both examples
are referenced throughout: `examples/toywalk` for the minimal shape and
`examples/forager` for anything involving vector state.

---

## 1. `spec.md`: write this first

The spec is not documentation. It is the artifact both implementations are
written from, and the only reason the differential test means anything. If
the spec is ambiguous, two readings of it will disagree and you will spend a
day deciding which one is "right" with nothing to appeal to.

Nail down every section. The scaffold TODO-marks all of them.

### State space

A table with **dtype and bounds** for every field.

```markdown
| field   | dtype       | bounds           | meaning                       |
|---------|-------------|------------------|-------------------------------|
| pos     | int64[2]    | [0, G-1] each    | agent cell, (x, y)            |
| berries | int64[K, 2] | [0, G-1] each    | berry cells, fixed per episode|
| alive   | bool[K]     | n/a              | berry not yet collected       |
| energy  | float32     | <= ENERGY_MAX    | remaining energy              |
| t       | int64       | [0, MAX_STEPS]   | in-episode step counter       |
```

**The step counter `t` must be part of state.** Random draws are keyed on it,
so both implementations have to agree on what it is at every moment.

Dtypes are load-bearing. An `int32` where the spec says `int64`, or a float64
division where it says float32, breaks bit-identity, and the differential
test reports it as a dtype mismatch rather than a value mismatch, which is a
much better error message than the one you would have got at 3am in week six.

### Actions

The encoding and its semantics. Be explicit about the mapping:

```markdown
| action | direction | delta (dx, dy) |
|--------|-----------|----------------|
| 0      | north     | (0, +1)        |
```

### Observations

The exact encoding: shape, dtype, and the **order of operations** where
float precision is involved. forager's spec says:

> Every division is computed **in float32**: cast the integer to float32, then
> divide by the float32 constant. This ordering is normative. It is what
> makes the two implementations bit-identical.

That sentence is doing real work. `float32(a) / float32(b)` and
`float32(a / b)` are not always the same number, and if the spec does not
choose, the two implementations will each choose differently.

### Rewards

The reward function, exactly, including the arithmetic order if any
accumulation is involved. Note which float computations genuinely cannot be
bit-identical between a scalar and a vectorized implementation. Those state
fields need `x-atol` in `schema.json` (see
[validation.md](validation.md#declaring-a-float-tolerance)).

Do not declare a tolerance pre-emptively. Write the spec so both sides do the
same operations in the same order, run the battery, and only add a tolerance
if you actually observe drift. forager was expected to need one for its
energy accumulation and turned out not to.

### Termination

Every condition. **Episodes must terminate under random actions**. The
auto-reset test needs to observe a termination, and it skips (loudly) if it
never sees one. A step cap alone is enough, but something that terminates
sooner makes the test faster and more thorough.

### Reset

The initial state distribution, and which RNG slots reset uses. Reset draws
use step 0 with their own dedicated slots.

### Invariants

An enumerated list of properties true in **every** state:

```markdown
1. `pos_in_bounds`: 0 <= pos[i] <= G - 1 for both coordinates.
2. `step_counter`: 0 <= t <= MAX_STEPS.
3. `fresh_episode`: t == 0 implies every alive[k] is True.
```

Each becomes an `@invariant` on the batched env, checked across the whole
batch after every step when `debug=True`. Write more than you think you need;
they are the cheapest bug-finding you will ever do.

### RNG slots

A table of every random decision. See
[rng-and-slots.md](rng-and-slots.md). This is the part that most repays
care.

---

## 2. `schema.json`: the serialized form

`$defs/state` is the canonical serialized state, and it is what the
differential test actually compares. It must mirror the spec's state table
exactly: same field names, same shapes, same bounds.

Factor repeated shapes with `$defs` and `$ref`, the way forager does with its
`cell` definition:

```json
"cell": {
  "type": "array",
  "items": { "type": "integer", "minimum": 0, "maximum": 7 },
  "minItems": 2, "maxItems": 2
}
```

`$defs/action` and `$defs/trajectory` are also required.
`test_spec_contract` fails if any are missing, and `test_replay` validates
real trajectory files against the trajectory definition.

---

## 3. `__init__.py`: constants and slots

Constants live here so both implementations import the same values rather
than each transcribing them from the spec:

```python
G = 8
K = 6
MAX_STEPS = 80
GUST_P = 0.15
START_ENERGY = np.float32(1.0)     # float32 constants stay float32


class Slots(IntEnum):
    """RNG slots. Mirrors the table in spec.md."""
    AGENT_X = 0
    BERRY_X = 2
    GUST = 5
```

Sharing constants is fine and desirable. Sharing *logic* is not. That is
what would make the differential test circular.

---

## 4. `reference.py`: obviously correct

Subclass `ReferenceEnv` and implement five methods.

```python
class ForagerReference(ReferenceEnv):
    def reset(self, seed, episode=0): ...      # -> State
    def step(self, action): ...                # -> (State, reward, terminated, info)
    def observe(self, state): ...              # -> np.ndarray (never a tensor)
    def to_json(self, state): ...              # -> dict matching $defs/state
    def from_json(self, obj): ...              # inverse of to_json
```

The style rules matter more than the code:

- **One instance.** No batching, no tensors, no vectorization.
- **Dataclass state, explicit control flow.** Every rule in `step` should be
  traceable to a spec line. Annotate them:

  ```python
  # spec: transition step 3. Clamp AFTER the rotation.
  ```

- **All randomness through `simulacrum.rng`.** Never `random`,
  `numpy.random` or `torch.rand`; those cannot be made bit-identical to the
  batched side.
- **Call `self.seed_episode(seed, episode)` at the top of `reset`.** It
  derives and stores `self.key`.
- **It does not auto-reset.** The caller observes `terminated` and calls
  `reset` again with an incremented `episode`. The harness bridges this to
  the batched convention.

Resist the urge to make it elegant. It exists to be checkable by eye.

---

## 5. `fast.py`: the batched rewrite

**Close `reference.py` before you start.** Write this from the spec.

Subclass `BatchedEnv` and implement four methods:

```python
class ForagerBatched(BatchedEnv):
    def _reset_instances(self, mask): ...   # init where mask is True, others bit-untouched
    def _step_impl(self, actions): ...      # -> (rewards[N], terminated[N])
    def observe(self): ...                  # -> [N, ...]
    def state_tensors(self): ...            # -> {schema field: Tensor[N, ...]}
```

The base class owns the step loop, RNG seeding state, auto-reset, terminal
state capture and invariant checking. Things to know:

- **`state_tensors` gives you serialization for free.** The default
  `slice_to_json` emits one flat field per entry, which is correct for both
  examples including forager's nested `[N, K, 2]` berries. Override it only
  for genuinely irregular state. And if you do, override
  `snapshot_slice_to_json` consistently, since terminal states are read from
  a pre-reset snapshot rather than the live tensors.
- **`_step_impl` must not touch `self.t`, `self.keys` or `self.episodes`.**
  The base class advances them.
- **The step-cap check uses `self.t + 1`,** because `self.t` increments after
  `_step_impl` returns.
- **`_reset_instances` must leave non-masked instances bit-untouched.**
  Allocate full-size tensors on the first call, then masked-fill:

  ```python
  if not hasattr(self, "pos"):
      self._alloc()
  self.pos = torch.where(mask.unsqueeze(-1), fresh_pos, self.pos)
  ```

Everything about translating scalar logic into masked tensor logic, and the
broadcasting traps that make it dangerous, is in
[vectorization-cookbook.md](vectorization-cookbook.md). Read it before
writing this file, not after.

### Invariants

One `@invariant` per spec entry, returning `Bool[N]`, True where it **holds**:

```python
@invariant("pos_in_bounds")
def _inv_pos(self):
    return ((self.pos >= 0) & (self.pos <= G - 1)).all(-1)
```

A violation under `debug=True` raises `InvariantViolation` naming the
invariant and the first failing instance, and dumps that instance's state to
`failures/` as renderable JSON.

---

## 6. `render.py`: how it looks

Two functions, consumed by the viz layer:

```python
def render_state_text(state: dict) -> str: ...
def render_state_mpl(state: dict, ax) -> None: ...
```

They take a **schema-conformant state dict**, not an env object. Renderers
contain zero game logic and never import `reference.py` or `fast.py`. See
[visualization.md](visualization.md).

---

## 7. `tests/conftest.py`: the harness config

One fixture describing your environment to the battery:

```python
@pytest.fixture
def harness_config():
    return HarnessConfig(
        name="forager",
        root=ENV_ROOT,
        reference_factory=ForagerReference,
        batched_factory=lambda n, debug=False: ForagerBatched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=4),
        benchmark_factory=lambda n, debug=False: ForagerBatched(
            n, debug=debug, compile=True, emit_final_states=False),
        scripted_policies=[...],
        min_speedup=8.0,
    )
```

Two options that are optional but strongly recommended:

**`benchmark_factory`**: the configuration you actually train with, usually
`compile=True` and `emit_final_states=False`. The battery bit-checks it
against the plain batched env, so the differential guarantee transfers to the
thing that really runs. Without it you have validated a configuration nobody
uses.

**`scripted_policies`**: a policy with a known return. These catch reward
bugs that bit-identity cannot, because a reward function can be wrong in both
implementations equally. Two policies that should differ are better than one:
forager asserts that a berry-seeking policy returns `+9.196` and a policy
that walks north forever returns `-4.404`, and the gap between them is the
real claim. If you cannot derive the number analytically, measure it once and
encode it as a regression gate, but say so in a comment.

Tuning knobs (`n_seeds`, `n_steps`, `sweep_batch`, `benchmark_batches`) are
documented on `HarnessConfig` in `simulacrum/harness/config.py`.

---

## 8. Validate

```bash
simulacrum validate myenv
```

Ten tests, a report, and a verdict. What each one proves is in
[validation.md](validation.md); what to do when one fails is in
[troubleshooting.md](troubleshooting.md).
