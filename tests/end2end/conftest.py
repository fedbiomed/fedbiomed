"""
Module for global PyTest configuration and fixtures

"""

import atexit
import os
import shutil
import socket
import tempfile

# Redirect the researcher component created on `fedbiomed.researcher.config`
# import to a temp dir, so tests never write it into the repository.
if "FBM_RESEARCHER_COMPONENT_ROOT" not in os.environ:
    _researcher_root = tempfile.mkdtemp(prefix="fbm-researcher-e2e-")
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = _researcher_root
    atexit.register(shutil.rmtree, _researcher_root, ignore_errors=True)

import pytest
from helpers import kill_registered_subprocesses, stop_researcher_server

os.environ["FBM_DEBUG"] = "1"


@pytest.fixture(scope="module")
def port():
    """Return an available port shared by the researcher and nodes."""
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return str(sock.getsockname()[1])


@pytest.fixture(scope="module", autouse=True)
def module_environment(request):
    """Expose a temporary directory for the module and guarantee its teardown.

    Every component the module creates lives under this directory, so removing
    it also removes what a failed test left behind. The finalizers run in
    reverse order and pytest reports all of them, so a failure to stop the
    server does not prevent the processes and the directory from being cleaned.
    """
    tmp_dir = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    pytest.temporary_test_directory = tempfile.TemporaryDirectory(
        prefix="fedbiomed-e2e-", dir=tmp_dir
    )
    print(f"##### FBM: Temporary test directory {pytest.temporary_test_directory.name}")
    print(f"\n#######  Running test {request.node}:{request.node.name} --------")

    request.addfinalizer(pytest.temporary_test_directory.cleanup)
    request.addfinalizer(kill_registered_subprocesses)
    request.addfinalizer(stop_researcher_server)

    yield
