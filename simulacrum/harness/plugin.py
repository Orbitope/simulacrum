"""pytest plugin wiring the battery to an environment package and persisting
the validation report.

Environment packages load it with one line in tests/conftest.py::

    pytest_plugins = ["simulacrum.harness.plugin"]

and provide a ``harness_config`` fixture returning a
:class:`simulacrum.harness.HarnessConfig`.
"""

from __future__ import annotations

import pytest

from simulacrum import spec as spec_mod
from simulacrum.harness import report as report_mod

_STASH = pytest.StashKey[dict]()


def pytest_configure(config):
    config.stash[_STASH] = {"results": {}, "extras": {}, "cfg_info": None}


@pytest.fixture
def schema_atol(harness_config):
    """Declared per-field x-atol map from schema.json's state definition."""
    return spec_mod.collect_atol(spec_mod.load_schema(harness_config.root))


@pytest.fixture
def record_extra(request):
    """Battery tests call record_extra(key, value) to persist structured data
    (throughput numbers, tolerances used) into validation_report.json."""
    stash = request.config.stash[_STASH]

    def _record(key: str, value) -> None:
        stash["extras"][key] = value

    return _record


@pytest.fixture(autouse=True)
def _register_env(request):
    """Capture the HarnessConfig (name/root) for the report writer. Autouse,
    but only touches harness_config on tests that already request it."""
    if "harness_config" in request.fixturenames:
        cfg = request.getfixturevalue("harness_config")
        request.config.stash[_STASH]["cfg_info"] = {"name": cfg.name, "root": str(cfg.root)}
    yield


def _short_failure(report) -> str | None:
    if report.outcome == "passed":
        return None
    if report.outcome == "skipped":
        try:
            return str(report.longrepr[2])
        except Exception:
            return str(report.longrepr)
    try:
        return str(report.longrepr.reprcrash.message)[:2000]
    except Exception:
        return str(report.longrepr)[:2000]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Record outcomes here (where we have item.config) rather than smuggling
    # the stash through an attribute on the report object — ad-hoc report
    # attributes don't survive serialization under pytest-xdist.
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        item.config.stash[_STASH]["results"][item.name] = {
            "outcome": report.outcome,
            "duration": round(report.duration, 3),
            "message": _short_failure(report),
        }


def pytest_sessionfinish(session, exitstatus):
    stash = session.config.stash.get(_STASH, None)
    if not stash or stash["cfg_info"] is None:
        return
    if getattr(session.config.option, "keyword", None):
        # A -k subset run must not overwrite the report: unrun required tests
        # would count as missing and clobber a passing report with FAIL.
        print("\n[simulacrum] subset run (-k): validation_report.json NOT updated "
              "— run the full battery to refresh the gate")
        return
    report_mod.write_report(stash["cfg_info"], stash["results"], stash["extras"])
