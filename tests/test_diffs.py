from simulacrum.harness.diffs import diff_states


def test_exact_equality():
    a = {"x": 1, "y": [1.5, 2.5], "z": {"w": True}}
    assert diff_states(a, {"x": 1, "y": [1.5, 2.5], "z": {"w": True}}, {}) == []


def test_int_mismatch_and_missing_fields():
    diffs = diff_states({"x": 1, "y": 2}, {"x": 3}, {})
    paths = {d.path for d in diffs}
    assert paths == {"x", "y"}


def test_float_bitwise_by_default():
    diffs = diff_states({"v": 0.1}, {"v": 0.1 + 1e-18}, {})
    assert not diffs  # below half-ulp: 0.1 + 1e-18 == 0.1 in float64
    import math
    diffs = diff_states({"v": 0.1}, {"v": math.nextafter(0.1, 1.0)}, {})
    assert len(diffs) == 1 and "no x-atol" in diffs[0].reason  # one ulp off fails


def test_declared_atol_allows_small_drift():
    atol = {"v": 1e-6}
    assert diff_states({"v": 0.1}, {"v": 0.1 + 1e-8}, atol) == []
    diffs = diff_states({"v": 0.1}, {"v": 0.1 + 1e-3}, atol)
    assert len(diffs) == 1 and "x-atol" in diffs[0].reason


def test_atol_on_array_items():
    atol = {"cards/*/value": 1e-6}
    ref = {"cards": [{"value": 1.0}, {"value": 2.0}]}
    fast = {"cards": [{"value": 1.0 + 1e-9}, {"value": 2.0}]}
    assert diff_states(ref, fast, atol) == []


def test_bool_int_not_confused():
    diffs = diff_states({"v": True}, {"v": 1}, {})
    assert len(diffs) == 1


def test_int_float_dtype_drift_caught():
    # 5 vs 5.0 is numerically equal but signals dtype drift in serialized
    # state — must be flagged, not silently passed.
    diffs = diff_states({"v": 5}, {"v": 5.0}, {})
    assert len(diffs) == 1 and "dtype drift" in diffs[0].reason
    diffs = diff_states({"v": 5.0}, {"v": 5}, {})
    assert len(diffs) == 1
    # ...and a declared x-atol does not excuse a type mismatch.
    diffs = diff_states({"v": 5}, {"v": 5.0}, {"v": 1e-6})
    assert len(diffs) == 1


def test_length_mismatch():
    diffs = diff_states({"xs": [1, 2]}, {"xs": [1]}, {})
    assert len(diffs) == 1 and "length" in diffs[0].reason
