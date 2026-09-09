import atexit
import os
import shutil
import tempfile

import pytest

# Default researcher workflow tests to debug-mode exception behavior.
os.environ.setdefault("FBM_DEBUG", "1")

# Redirect the researcher component created on `fedbiomed.researcher.config`
# import to a temp dir, so tests never write it into the repository.
if "FBM_RESEARCHER_COMPONENT_ROOT" not in os.environ:
    _researcher_root = tempfile.mkdtemp(prefix="fbm-researcher-test-")
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = _researcher_root
    atexit.register(shutil.rmtree, _researcher_root, ignore_errors=True)


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
