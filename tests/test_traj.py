import json

import pytest

from simulacrum.traj.picker import random_sample, split_episodes, worst_return
from simulacrum.traj.reader import read_trajectory
from simulacrum.traj.writer import TrajectoryWriter


def _make(tmp_path, name, rewards, suffix=".json"):
    w = TrajectoryWriter("dummy", 1, {"x": 0}, source="reference")
    for t, r in enumerate(rewards):
        w.record(t, 0, {"x": t + 1}, 0, r, t == len(rewards) - 1)
    return w.write(tmp_path / f"{name}{suffix}")


def test_roundtrip_json_and_jsonl(tmp_path):
    for suffix in (".json", ".jsonl"):
        path = _make(tmp_path, f"t{suffix.strip('.')}", [1.0, -2.0, 3.0], suffix)
        traj = read_trajectory(path)
        assert traj.metadata["env"] == "dummy"
        assert traj.initial["state"] == {"x": 0}
        assert [s["reward"] for s in traj.steps] == [1.0, -2.0, 3.0]
        assert traj.total_return == 2.0
        assert traj.steps[-1]["terminated"] is True


def test_reader_validates_against_schema(tmp_path):
    schema = {"$defs": {
        "state": {"type": "object", "properties": {"x": {"type": "integer"}},
                  "required": ["x"], "additionalProperties": False},
        "action": {"type": "integer"},
        "trajectory": {"type": "object", "properties": {
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "state": {"$ref": "#/$defs/state"}}}}}},
    }}
    path = _make(tmp_path, "ok", [1.0])
    read_trajectory(path, schema=schema)  # passes

    bad = json.loads(path.read_text())
    bad["steps"][0]["state"]["x"] = "not-an-int"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        read_trajectory(bad_path, schema=schema)


def test_worst_return_and_random_sample(tmp_path):
    _make(tmp_path, "good", [5.0, 5.0])
    _make(tmp_path, "bad", [-10.0])
    _make(tmp_path, "mid", [0.0])
    worst = worst_return(tmp_path, k=2)
    assert [p.stem for p in worst] == ["bad", "mid"]
    assert len(random_sample(tmp_path, k=2, seed=0)) == 2


def test_split_episodes(tmp_path):
    w = TrajectoryWriter("dummy", 1, {"x": 0})
    pattern = [False, False, True, False, True, False]
    for t, term in enumerate(pattern):
        w.record(t, 0, {"x": t}, 0, 0.0, term)
    eps = split_episodes(w.traj)
    assert [len(e) for e in eps] == [3, 2, 1]
    assert eps[0][-1]["terminated"] and not eps[2][-1]["terminated"]
