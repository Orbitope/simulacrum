import pytest

from simulacrum import spec as spec_mod
from simulacrum.scaffold import scaffold


def test_scaffold_layout_and_names(tmp_path):
    root = scaffold("my_env", tmp_path)
    for rel in ["spec.md", "schema.json", "__init__.py", "reference.py",
                "fast.py", "render.py", "tests/conftest.py", "tests/test_battery.py"]:
        assert (root / rel).exists(), rel
    assert "MyEnvReference" in (root / "reference.py").read_text()
    assert "from my_env import Slots" in (root / "reference.py").read_text()


def test_scaffold_rejects_bad_names(tmp_path):
    for bad in ["My-Env", "9lives", "CamelCase", "with space"]:
        with pytest.raises(ValueError):
            scaffold(bad, tmp_path)


def test_scaffold_refuses_overwrite(tmp_path):
    scaffold("dupe", tmp_path)
    with pytest.raises(FileExistsError):
        scaffold("dupe", tmp_path)


def test_scaffolded_spec_fails_contract_until_filled(tmp_path):
    # The TODO-marked spec must NOT pass the contract check — the gate should
    # force authors to actually write the spec.
    root = scaffold("todoenv", tmp_path)
    with pytest.raises(spec_mod.SpecError):
        spec_mod.load_spec(root)
    # The scaffolded schema is at least structurally valid JSON Schema.
    spec_mod.load_schema(root)


def test_export_pack_shape(tmp_path):
    import json

    from simulacrum.traj.writer import TrajectoryWriter
    from simulacrum.viz.export_pack import export_pack

    w = TrajectoryWriter("dummy", 1, {"x": 0, "name": "spot"})
    w.record(0, 0, {"x": 1, "name": "spot"}, 0, -1.0, False)
    w.record(1, 0, {"x": 2, "name": "spot"}, 1, 10.0, True)
    tpath = w.write(tmp_path / "run.json")

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"$defs": {
        "state": {"type": "object"}, "action": {}, "trajectory": {"type": "object"}}}))

    pack = export_pack([tpath], schema_path, tmp_path / "pack")
    manifest = json.loads((pack / "manifest.json").read_text())
    assert manifest["trajectories"][0]["total_return"] == 9.0
    flat = json.loads((pack / "run.flat.json").read_text())
    # numeric column captured; non-scalar (string) field skipped
    assert [c["name"] for c in flat["state_fields"]] == ["x"]
    assert flat["skipped_fields"] == ["name"]
    assert flat["state_fields"][0]["values"] == [0.0, 1.0, 2.0]
    assert flat["series"][1]["values"] == [-1.0, 9.0]
    assert flat["terminal_step_indices"] == [1]
