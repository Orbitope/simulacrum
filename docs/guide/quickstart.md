# Quickstart

Ten minutes from nothing to a passing validation report.

## Install

```bash
git clone https://github.com/orbitope/simulacrum
cd simulacrum
pip install -e .
```

Requires Python 3.10+, PyTorch 2.2+, NumPy, jsonschema and pytest. Matplotlib
is optional and only needed for figure rendering (`pip install -e '.[viz]'`).

## Run the examples first

Before writing anything, watch the battery run on something that already
works:

```bash
simulacrum validate examples/toywalk
```

The full battery takes a minute or two — most of it is `torch.compile` warmup
in the last two tests. You should end with:

```
  test_spec_contract              PASS      0.01s
  test_differential               PASS      1.83s
  test_batch_independence         PASS      1.69s
  ...
  overall: PASS — eligible for training
```

`examples/forager` is the bigger example and is worth running too. Both write
a `validation_report.json` next to the environment package. That file is
gitignored on purpose — see [validation.md](validation.md#the-report-is-not-a-build-artifact).

## Scaffold your own

```bash
simulacrum new myenv
```

That writes:

```
myenv/
  spec.md           # the single source of truth — TODO-marked
  schema.json       # state/action schemas; trajectory def prefilled
  __init__.py       # constants and the Slots IntEnum
  reference.py      # ReferenceEnv stub
  fast.py           # BatchedEnv stub
  render.py         # rendering hooks for the viz layer
  tests/
    conftest.py     # the harness_config fixture
    test_battery.py # the entire battery, one star-import
```

`tests/test_battery.py` is one line and you never edit it:

```python
from simulacrum.harness.battery import *  # noqa: F401,F403
```

## The order matters

This is the part people skip, and skipping it removes the entire point of the
framework.

1. **Write `spec.md` first.** State fields with dtypes and bounds, action
   encoding, the exact observation encoding, the reward function, termination
   rules, an enumerated invariant list, and an enumerated RNG-slot table.
2. **Fill in `schema.json`** to match the spec's state table exactly.
3. **Write `reference.py` from the spec.** Readable, single-instance, explicit
   `if`s. Every line should be traceable to a spec line.
4. **Write `fast.py` from the spec** — not from `reference.py`. Do not have it
   open. If you port `reference.py` line by line you will port its bugs too,
   and the differential test will happily confirm that both copies of the same
   mistake agree with each other.
5. **`simulacrum validate myenv`**, and fix what it finds.

[writing-an-environment.md](writing-an-environment.md) walks through each of
those steps in detail.

## While you are iterating

Run a subset instead of the whole battery:

```bash
simulacrum validate myenv -k differential
simulacrum validate myenv -k "differential or batch_independence"
```

Subset runs deliberately **do not** update `validation_report.json` — a report
must reflect a full battery run or it is not a gate. The CLI tells you so.

## Wire the gate into training

At the top of any training script:

```python
from simulacrum.harness import require_fresh_report

require_fresh_report("path/to/myenv", strict=True)
```

That raises if the report is missing, failing, or older than the
environment's source files. Without `strict=True` it prints a loud banner and
returns `False`, which is the right choice for exploratory work and the wrong
one for a run you intend to trust.

## Where to go next

- [writing-an-environment.md](writing-an-environment.md) — the full walkthrough
- [rng-and-slots.md](rng-and-slots.md) — designing your random draws
- [troubleshooting.md](troubleshooting.md) — when the battery says no
