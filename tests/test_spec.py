import pytest

from simulacrum import spec as spec_mod

FULL_SPEC = """# myenv spec

## State space
| x | int64 | [0, 10] |

## Actions
0 = left, 1 = right.

## Observations
float32[1].

## Reward function
-1 per step.

## Termination
x == 10.

## Reset
x = 0.

## Invariants
1. x in bounds.

## RNG slots
| 0 | SLIP | every step | bernoulli |
"""


def _write(tmp_path, text):
    (tmp_path / "spec.md").write_text(text)
    return tmp_path


def test_full_spec_passes(tmp_path):
    spec_mod.load_spec(_write(tmp_path, FULL_SPEC))


def test_singular_headings_accepted(tmp_path):
    # "## Reward function" (no plural 'rewards') must satisfy the contract.
    sections = spec_mod.load_spec(_write(tmp_path, FULL_SPEC))
    assert "reward function" in sections


def test_code_fences_do_not_create_sections_or_todos(tmp_path):
    text = FULL_SPEC + """
## Notes

```python
# TODO: this is an illustrative snippet, not a spec placeholder
# INIT_POSITION = 0
```
"""
    sections = spec_mod.load_spec(_write(tmp_path, text))
    assert "init_position = 0" not in sections


def test_prose_todo_still_gates(tmp_path):
    text = FULL_SPEC.replace("x = 0.", "TODO: define the reset distribution")
    with pytest.raises(spec_mod.SpecError, match="TODO"):
        spec_mod.load_spec(_write(tmp_path, text))


def test_todo_substring_in_word_does_not_gate(tmp_path):
    text = FULL_SPEC.replace("x = 0.", "No TODOs remain in this spec.")
    spec_mod.load_spec(_write(tmp_path, text))


def test_missing_section_reported(tmp_path):
    text = FULL_SPEC.replace("## Invariants", "## Properties")
    with pytest.raises(spec_mod.SpecError, match="invariant"):
        spec_mod.load_spec(_write(tmp_path, text))