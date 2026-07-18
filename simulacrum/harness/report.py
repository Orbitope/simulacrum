"""The validation report artifact: written by the battery, read by training.

Validation is a gate, not a suggestion. ``validation_report.json`` in the
environment package root records what passed, when, at which git SHA, and how
fast the env runs. Training scripts call :func:`require_fresh_report` and
refuse (or at minimum warn loudly) when the report is missing, failing, or
older than the environment's source files.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path

REPORT_FILENAME = "validation_report.json"

# Tests that must PASS (not skip) for an env to be training-eligible.
REQUIRED_TESTS = (
    "test_spec_contract",
    "test_differential",
    "test_batch_independence",
    "test_invariant_sweep",
    "test_determinism",
    "test_replay",
)


def write_report(cfg_info: dict, results: dict, extras: dict) -> Path:
    import numpy
    import torch

    import simulacrum
    from simulacrum.traj.writer import _git_sha

    root = Path(cfg_info["root"])
    outcomes = {name: r["outcome"] for name, r in results.items()}
    failures = [n for n, o in outcomes.items() if o == "failed"]
    missing = [n for n in REQUIRED_TESTS if outcomes.get(n) != "passed"]
    overall_pass = not failures and not missing

    report = {
        "env": cfg_info["name"],
        "overall_pass": overall_pass,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(root),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "simulacrum": simulacrum.__version__,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "tests": results,
        "required_tests_not_passed": missing,
        **extras,
    }
    path = root / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2))
    return path


def print_summary(report: dict) -> None:
    name_w = max((len(n) for n in report["tests"]), default=10) + 2
    print(f"\nValidation report: {report['env']}")
    print(f"  created: {report['created_at']}   git: {report.get('git_sha') or 'n/a'}")
    print("  " + "-" * (name_w + 30))
    for name, r in report["tests"].items():
        mark = {"passed": "PASS", "failed": "FAIL", "skipped": "skip"}.get(r["outcome"], "?")
        line = f"  {name:<{name_w}} {mark:<6} {r['duration']:>7.2f}s"
        if r["outcome"] == "skipped" and r.get("message"):
            line += f"   ({r['message'][:60]})"
        print(line)
    tp = report.get("throughput")
    if tp:
        print("  " + "-" * (name_w + 30))
        print(f"  reference: {tp['reference_steps_per_sec']:>12,.0f} steps/s")
        for label, sps in tp["batched"].items():
            print(f"  batched {label:<18} {sps:>12,.0f} steps/s")
        print(f"  speedup at largest batch: {tp['speedup_at_largest_batch']:.0f}x")
    tol = report.get("tolerance_fields_used")
    if tol:
        print(f"  tolerance fields exercised: {tol}")
    missing = report.get("required_tests_not_passed")
    if missing:
        print(f"  required tests not passed: {missing}")
    verdict = "PASS — eligible for training" if report["overall_pass"] else "FAIL — NOT eligible for training"
    print(f"  overall: {verdict}\n")


def _source_mtime(env_root: Path) -> float:
    newest = 0.0
    for pattern in ("*.py", "spec.md", "schema.json"):
        for p in env_root.rglob(pattern):
            if "failures" in p.parts or "__pycache__" in p.parts:
                continue
            newest = max(newest, p.stat().st_mtime)
    return newest


def require_fresh_report(env_root: str | Path, *, strict: bool = False) -> bool:
    """Check the environment has a fresh, passing validation report.

    Call this at the top of training scripts. Returns True when the gate is
    satisfied. Otherwise warns LOUDLY on stderr (or raises, with
    ``strict=True``): report missing, report failing, or environment source
    files modified after the report was written.
    """
    env_root = Path(env_root)
    path = env_root / REPORT_FILENAME

    def fail(msg: str) -> bool:
        banner = (
            "\n" + "!" * 72 + f"\n!! VALIDATION GATE: {msg}\n"
            f"!! Run: simulacrum validate {env_root}\n" + "!" * 72 + "\n"
        )
        if strict:
            raise RuntimeError(banner)
        print(banner, file=sys.stderr)
        return False

    if not path.exists():
        return fail(f"no {REPORT_FILENAME} found for env at {env_root}")
    report = json.loads(path.read_text())
    if not report.get("overall_pass"):
        return fail(f"validation report for '{report.get('env')}' is FAILING")
    created = datetime.datetime.fromisoformat(report["created_at"]).timestamp()
    if _source_mtime(env_root) > created:
        return fail(
            f"env source files modified after the validation report was written "
            f"({report['created_at']}) — report is STALE")
    return True
