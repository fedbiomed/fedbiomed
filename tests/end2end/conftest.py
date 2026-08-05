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
from helpers import (
    create_federation,
    kill_registered_subprocesses,
    stop_researcher_server,
)

os.environ["FBM_DEBUG"] = "1"


@pytest.fixture(scope="module")
def port():
    """Return an available port shared by the researcher and nodes."""
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return str(sock.getsockname()[1])


@pytest.fixture(scope="module", autouse=True)
def e2e_workspace(request):
    """Directory holding everything the module creates, removed afterwards.

    The finalizers run in reverse order and pytest reports all of them, so a
    failure to stop the server does not prevent the processes and the directory
    from being cleaned.
    """
    tmp_dir = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    workspace = tempfile.TemporaryDirectory(prefix="fedbiomed-e2e-", dir=tmp_dir)
    print(f"##### FBM: Workspace {workspace.name}")
    print(f"\n#######  Running test {request.node}:{request.node.name} --------")

    request.addfinalizer(workspace.cleanup)
    request.addfinalizer(kill_registered_subprocesses)
    request.addfinalizer(stop_researcher_server)

    yield workspace.name


@pytest.fixture(scope="module")
def federation(e2e_workspace, port):
    """The researcher and nodes the module runs against."""
    with create_federation(e2e_workspace, port) as federation_:
        yield federation_
