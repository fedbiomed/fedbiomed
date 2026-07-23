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

import psutil
import pytest
from helpers import (
    CONFIG_PREFIX,
    kill_process,
    stop_researcher_server,
)

_PORT = 50151

os.environ["FBM_DEBUG"] = "1"


@pytest.fixture(scope="module")
def port():
    """Increases and return port for researcher server"""
    global _PORT

    _PORT += 1
    return str(_PORT)


@pytest.fixture(scope="module", autouse=True)
def data():
    """Create and expose the shared temporary directory for the test module."""
    home_dir = os.path.expanduser("~")
    tmp_dir = os.path.join(home_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"##### FBM: Setting temporary test directory to {tmp_dir}")
    pytest.temporary_test_directory = tempfile.TemporaryDirectory(dir=tmp_dir)


@pytest.fixture(scope="module", autouse=True)
def post_session(request, data):
    """This method makes sure that the environment is clean to execute another test"""

    print("#### Killing e2e processes before executing test module")
    kill_e2e_test_processes()
    print("#### Killing is completed --------")
    print(f"\n#######  Running test {request.node}:{request.node.name} --------")

    yield

    stop_researcher_server()
    print("#### Killing e2e processes after the tests -----")
    kill_e2e_test_processes()
    print("#### Killing is completed")
    print("\n###### Cleaning temporary directory: started -----\n")
    print(f"Directory: {pytest.temporary_test_directory}")
    pytest.temporary_test_directory.cleanup()
    # Remove ~/_tmp only if empty: it may hold data not owned by the tests
    tmp_dir = os.path.join(os.path.expanduser("~"), "_tmp")
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    print("\n###### Cleaning temporary directory: finished  -----\n\n")
    print(
        f"#### Module tests have finished {request.node}:{request.node.name} --------"
    )


def kill_e2e_test_processes():
    """Kills end2end processes if any existing"""

    for process in psutil.process_iter():
        try:
            cmdline = process.cmdline()
        except psutil.Error as e:
            print(f"\n #####: FBM: PSUTIL ERROR: {e}")
            continue
        else:
            if any([re.search(rf"^{CONFIG_PREFIX}.*\.ini$", cmd) for cmd in cmdline]):
                print(f'#####: FBM: Found a processes not killed: "{cmdline}"')
                kill_process(process)
