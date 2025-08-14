"""
Module for global PyTest configuration and fixtures

"""

import re
import os
import tempfile
import shutil

import pytest
import psutil

from helpers import (
    kill_process,
    CONFIG_PREFIX,
)

_PORT = 50151


@pytest.fixture(scope="module")
def port():
    """Increases and return port for researcher server"""
    global _PORT

    _PORT += 1
    return str(_PORT)


@pytest.fixture(scope="module", autouse=True)
def data():
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

    print("#### Kiling e2e processes after the tests -----")
    kill_e2e_test_processes()
    print("#### Killing is completed")
    print("\n###### Cleaning temporary directory: started -----\n")
    print(f"Directory: {pytest.temporary_test_directory}")
    pytest.temporary_test_directory.cleanup()
    tmp_dir = os.path.join(os.path.expanduser("~"), "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n###### Cleaning temporary directory: finished  -----\n\n")
    print(
        f"#### Module tests have finished {request.node}:{request.node.name} --------"
    )


def kill_e2e_test_processes():
    """Kills end2end processeses if any existing"""

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
