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
    clear_component_data
)

from fedbiomed.common.constants import ComponentType, ComponentType
from fedbiomed.common.utils import CONFIG_DIR

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



    # Clear remaining component data if existing
    configs = glob.glob( os.path.join( CONFIG_DIR, f"{CONFIG_PREFIX}*.ini"))

    for path in configs:
        cfg = configparser.ConfigParser()
        cfg.read(path)
        component_type = cfg.get('default', 'component')
        print(f"Clearing remaining component files  for {path}")
        if component_type == ComponentType.NODE.name:
            config = importlib.import_module("fedbiomed.node.config").NodeConfig
        elif component_type == ComponentType.RESEARCHER.name:
            config = importlib.import_module("fedbiomed.researcher.config").ResearcherConfig
        config = config(name=os.path.basename(path))
        clear_component_data(config)

        del cfg
        del config


    print('#### Checking processes has fininshed.')
    print('Module tests have finished --------------------------------------------')



