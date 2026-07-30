"""Skip hook and shared fixtures for the Phase 4.5 benchmarks.

The `--bench` flag itself is declared in `tests/conftest.py`; pytest
only calls `pytest_addoption` on initial conftests, so declaring it
here would leave the option unknown when the suite is run as
`pytest tests/chemistry`.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip everything in this directory unless `--bench` was passed."""
    if config.getoption("--bench"):
        return
    skip = pytest.mark.skip(reason="benchmark; pass --bench to run")
    here = Path(__file__).parent.resolve()
    for item in items:
        if here in Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def results_dir():
    """Path to `tests/chemistry/perf/results/` (created if absent)."""
    rdir = Path(__file__).parent.resolve() / "results"
    rdir.mkdir(exist_ok=True)
    return rdir
