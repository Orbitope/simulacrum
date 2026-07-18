"""simulacrum CLI: ``simulacrum new <name>``, ``simulacrum validate <path>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_new(args: argparse.Namespace) -> int:
    from simulacrum.scaffold import scaffold

    root = scaffold(args.name, args.dest)
    print(f"Scaffolded environment package at {root}")
    print("Next steps:")
    print(f"  1. Fill in {root}/spec.md (state space, actions, rewards, termination,")
    print("     INVARIANTS, RNG slots) and schema.json — the spec comes first.")
    print("  2. Write reference.py from the spec (readable, single-instance).")
    print("  3. Write fast.py from the spec (batched, masked) — not from reference.py.")
    print(f"  4. simulacrum validate {root}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import time

    import pytest

    from simulacrum.harness.report import REPORT_FILENAME, print_summary

    env_root = Path(args.path).resolve()
    tests_dir = env_root / "tests"
    if not tests_dir.is_dir():
        print(f"error: {tests_dir} not found — not an environment package?", file=sys.stderr)
        return 2

    started_at = time.time()
    pytest_args = [str(tests_dir), "-q", "--no-header"]
    if args.k:
        pytest_args += ["-k", args.k]
    exit_code = pytest.main(pytest_args)

    # Only trust a report THIS run wrote — if the battery errored before the
    # session-finish hook (e.g. a collection error), a stale report from an
    # earlier run may still be on disk.
    report_path = env_root / REPORT_FILENAME
    if report_path.exists() and report_path.stat().st_mtime >= started_at:
        report = json.loads(report_path.read_text())
        print_summary(report)
        print(f"Report written to {report_path}")
        return 0 if report["overall_pass"] else 1
    print(f"error: battery did not produce a fresh {report_path} "
          f"(it likely errored before running — see pytest output above)", file=sys.stderr)
    return exit_code or 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="simulacrum",
        description="Batched RL environment development framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a new environment package")
    p_new.add_argument("name", help="package name (lowercase identifier)")
    p_new.add_argument("--dest", default=".", help="parent directory (default: .)")
    p_new.set_defaults(func=_cmd_new)

    p_val = sub.add_parser("validate", help="run the validation battery on an env package")
    p_val.add_argument("path", help="environment package root")
    p_val.add_argument("-k", default=None, help="pytest -k expression (subset of battery)")
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
