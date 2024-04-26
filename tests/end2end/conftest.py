"""
Module for global PyTest configuration and fixtures

"""

import re
import uuid
import pytest
import psutil


from helpers import  (
    kill_process,
    CONFIG_PREFIX
)

from fedbiomed.common.constants import ComponentType

_PORT = 50052

@pytest.fixture(scope='session')
def port():
    return str(_PORT + 1)


@pytest.fixture(scope='module', autouse=True)
def post_session(request):
    """This method makes sure that the environment is clean to execute another test"""
    print(f"\n #########  Running test {request.node}:{request.node.name} ---------------------------")

    yield

    print("#### Checking remaining processes if there is any not killed after the tests")
    for process in psutil.process_iter():
        try:
            cmdline = process.cmdline()
        except psutil.Error:
            continue
        else:
            if any([re.search(fr'^{CONFIG_PREFIX}.*\.ini$', cmd) for cmd in cmdline]):
                print(f'Found a processes not killed: "{cmdline}"')
                kill_process(process)

    print('#### Checking processes has fininshed.')
    print('Module tests have finished --------------------------------------------')


