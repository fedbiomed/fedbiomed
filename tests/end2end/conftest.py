"""
Module for global PyTest configuration and fixtures

"""

import re
import os
import glob
import importlib

import pytest
import psutil
import configparser

from helpers import  (
    kill_process,
    CONFIG_PREFIX,
    clear_component_data,
    temporary_test_directory
)

from fedbiomed.common.constants import ComponentType, ComponentType
from fedbiomed.common.utils import CONFIG_DIR


_PORT = 50052

@pytest.fixture(scope='module')
def port():
    """Increases and return port for researcher server"""
    global _PORT

    _PORT += 1
    return str(_PORT)


@pytest.fixture(scope='module', autouse=True)
def post_session(request):
    """This method makes sure that the environment is clean to execute another test"""

    kill_e2e_test_processes()
    print(f"\n #########  Running test {request.node}:{request.node.name} --------")

    yield

    kill_e2e_test_processes()
    temporary_test_directory.cleanup()
    print('Module tests have finished --------------------------------------------')


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

