"""
Module for global PyTest configuration and fixtures

"""

import re
import os
import glob
import importlib
import tempfile

import pytest
import psutil

from helpers import  (
    kill_process,
    CONFIG_PREFIX,
)



_PORT = 50052

@pytest.fixture(scope='module')
def port():
    """Increases and return port for researcher server"""
    global _PORT

    _PORT += 1
    return str(_PORT)

@pytest.fixture(scope='module', autouse=True)
def data():
    pytest.temporary_test_directory = tempfile.TemporaryDirectory()

@pytest.fixture(scope='module', autouse=True)
def post_session(request, data):
    """This method makes sure that the environment is clean to execute another test"""

    kill_e2e_test_processes()
    print(f"\n #########  Running test {request.node}:{request.node.name} --------")

    yield

    kill_e2e_test_processes()
    print("\n\n ######### Cleaning temprorary directory: started -----\n\n")
    pytest.temporary_test_directory.cleanup()
    print("\n\n ######### Cleaning temprorary directory: finished  -----\n\n")
    print(f'Module tests have finished {request.node}:{request.node.name} --------')


def kill_e2e_test_processes():
    """Kills end2end processeses if any existing"""

    for process in psutil.process_iter():
        try:
            cmdline = process.cmdline()
        except psutil.Error:
            continue
        else:
            if any([re.search(fr'^{CONFIG_PREFIX}.*\.ini$', cmd) for cmd in cmdline]):
                print(f'Found a processes not killed: "{cmdline}"')
                kill_process(process)

