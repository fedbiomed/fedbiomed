import os

import pytest

# Default researcher workflow tests to debug-mode exception behavior.
os.environ.setdefault("FBM_DEBUG", "1")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Redirect the researcher component to pytest's temp area.

    `fedbiomed.researcher.config` creates the component when it is imported,
    which happens as test modules are collected - after this hook - so the
    redirection is in place by then and tests never write it into the
    repository. The directory comes from the factory behind `tmp_path`, so it
    sits with the rest of the run and is rotated with it; `trylast` lets the
    plugin owning that factory configure first.
    """
    if "FBM_RESEARCHER_COMPONENT_ROOT" in os.environ:
        return

    root = config._tmp_path_factory.mktemp("fbm-researcher")
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = str(root)


@pytest.fixture(autouse=True)
def _isolated_database(monkeypatch):
    """Give every test its own database.

    `TinyDBConnector` is a singleton that ignores the path it is handed, so
    without this every table in the session shares the file opened first. A test
    that leaves it half-built also breaks every test that opens a table later.
    """
    # Imported here so the environment set above applies to the import.
    from fedbiomed.common.db import TinyDBConnector

    monkeypatch.setattr(TinyDBConnector, "_instance", None)
