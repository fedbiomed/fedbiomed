"""
Module for global PyTest configuration and fixtures

"""

import atexit
import os
import re
import shutil
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
def port(free_tcp_port_factory):
    """Return an available port shared by the researcher and nodes."""
    return str(free_tcp_port_factory())


@pytest.fixture(scope="module", autouse=True)
def data():
    """Create and expose the shared temporary directory for the test module."""
    run_id = os.environ.get("FBM_E2E_RUN_ID", f"local-{os.getpid()}")
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    tmp_dir = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
    print(f"##### FBM: Setting temporary test directory under {tmp_dir}")
    pytest.temporary_test_directory = tempfile.TemporaryDirectory(
        prefix=f"fedbiomed-e2e-{run_id}-",
        dir=tmp_dir,
    )


@pytest.fixture(scope="module", autouse=True)
def post_session(request, data):
    """This method makes sure that the environment is clean to execute another test"""

    print("#### Killing e2e processes before executing test module")
    kill_e2e_test_processes()
    print("#### Killing is completed --------")
    print(f"\n#######  Running test {request.node}:{request.node.name} --------")

    try:
        yield
    finally:
        try:
            stop_researcher_server()
        finally:
            print("#### Killing e2e processes after the tests -----")
            try:
                kill_e2e_test_processes()
                print("#### Killing is completed")
            finally:
                print("\n###### Cleaning temporary directory: started -----\n")
                print(f"Directory: {pytest.temporary_test_directory}")
                pytest.temporary_test_directory.cleanup()
                print("\n###### Cleaning temporary directory: finished  -----\n\n")
                print(
                    "#### Module tests have finished "
                    f"{request.node}:{request.node.name} --------"
                )


def kill_e2e_test_processes():
    """Kills end2end processes if any existing"""
    kill_registered_subprocesses()
