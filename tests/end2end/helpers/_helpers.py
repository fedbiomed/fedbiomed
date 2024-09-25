"""
Helper methods for end2end tests
"""

import platform
import importlib
import shutil
import tempfile
import json
import asyncio
import os
import uuid
import threading
import multiprocessing
import subprocess
import functools

from contextlib import contextmanager
from typing import Dict, Any, Tuple, Callable, List

import pytest

from fedbiomed.common.constants import TENSORBOARD_FOLDER_NAME, ComponentType
from fedbiomed.common.config import Config
from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR

from ._execution import (
    fork_process,
    shell_process,
    fedbiomed_run,
    execute_in_paralel,
)
from .constants import CONFIG_PREFIX, FEDBIOMED_SCRIPTS, End2EndError



class PytestThread(threading.Thread):
    """Extension of Thread for PyTest to be able to fail thread properly"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            super().run()
        except BaseException as e:
            self.exception = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.exception:
            raise self.exception


def add_dataset_to_node(
    config: Config,
    dataset: dict
) -> True:
    """Adds given dataset using given configuration of the node"""

    tempdir_ = tempfile.TemporaryDirectory()
    d_file = os.path.join(tempdir_.name, "dataset.json")
    with open(d_file, "w", encoding="UTF-8") as file:
        json.dump(dataset, file)

    command = ["node", "--config", config.name, "dataset", "add", "--file", d_file]
    _ = fedbiomed_run(command, wait=True, on_failure=default_on_failure)
    tempdir_.cleanup()

    return True


def default_on_failure(process: subprocess.Popen):
    """Default function to execute when the process is on exit"""
    print(f"On failure callback: Process has failed!, {process}")
    raise End2EndError(f"Porcesses has failed! command: {process.args}")


def start_nodes(
    configs: list[Config],
    interrupt_all_on_fail: bool = True,
    on_failure: Callable = default_on_failure
) -> multiprocessing.Process:
    """Starts the nodes by given list of configs

    Args:
        configs: List of node config objects
    """

    processes = []
    for c in configs:
        # Keep it for debugging purposes
        if 'fail' in c.name:
            processes.append(
                fedbiomed_run(
                    ['node', "--config", c.name, 'unkown-commnad'], pipe=False))
        else:
            processes.append(
                fedbiomed_run(
                    ["node", "--config", c.name, "start"], pipe=False))

    t = PytestThread(
        target=execute_in_paralel,
        kwargs={
            'processes': processes,
            'interrupt_all_on_fail': interrupt_all_on_fail,
            'on_failure': on_failure})

    t.daemon = True
    t.start()

    return processes, t


def execute_script(file: str, activate: str = 'researcher'):
    """Executes given scripts"""

    if not os.path.isfile(file):
        raise End2EndError("file is not existing")

    if file.endswith('.py'):
        return execute_python(file, activate)

    if file.endswith('.ipynb'):
        return execute_ipython(file, activate)

    raise End2EndError('Unsopported file file. Please use .py or .ipynb')


def execute_python(file: str, activate: str):
    """Executes given python file in a process"""

    return shell_process(
        command=["python", f'{file}'],
        activate=activate,
        wait=True,
        on_failure=default_on_failure
    )

def execute_ipython(file: str, activate: str):
    """Executes given ipython file in a process"""

    return shell_process(
        command=["ipython", "-c", f'"%run {file}"'],
        activate=activate,
        wait=True,
        on_failure=default_on_failure
    )


def clear_component_data(config: Config):
    """Clears component related file"""

    # extract Component Type from config file

    _component_type = config.get('default', 'component')

    if _component_type == ComponentType.NODE.name:

        clear_node_data(config)

    elif _component_type == ComponentType.RESEARCHER.name:

        clear_researcher_data(config)


def clear_node_data(config: Config):
    """Clears data relative to Node, such as configuration file, database,
    node state

    Args:
        config: configuration object of the Node
    """
    node_id = config.get('default', 'id')
    # remove node's state
    _node_state_dir = os.path.join(VAR_DIR, f"node_state_{node_id}")

    if os.path.lexists(_node_state_dir):
        print("[INFO] Removing folder ", _node_state_dir)
        shutil.rmtree(_node_state_dir)

    # remove node's taskqueue
    _task_queue_dir = os.path.join(VAR_DIR,
                                    f'queue_manager_{node_id}')
    if os.path.lexists(_task_queue_dir):
        print("[INFO] Removing folder ", _task_queue_dir)
        shutil.rmtree(_task_queue_dir)

    # remove node's ssl material
    _cert_material_files = ('private_key', 'public_key')
    for cert_file in _cert_material_files:
        cert_material = config.get('certificate', cert_file,)
        _material_to_remove = os.path.join(CONFIG_DIR, cert_material)
        _material_to_remove_folder = os.path.dirname(_material_to_remove)
        if not os.path.lexists(_material_to_remove_folder):
            continue
        print("[INFO] Removing folder ", _material_to_remove_folder)
        shutil.rmtree(_material_to_remove_folder)  # remove the whole folder of cert

    # remove database
    _database_file_path = config.get('default', 'db')

    os.remove(os.path.join(VAR_DIR, _database_file_path))
    # remove Node's config file
    _clear_config_file_component(config)


def clear_researcher_data(config: Config):
    """Clears data relative to Researcher, mainly Researcher database, Researcher configuration file
    and files relative to secure aggregation and certificates

    Args:
        config: configuration object of Researcher
    """
    # ATTENTION: breakpoints should be removed when using
    # Experiment cleaning methods

    # remove Researcher database (should be done before deleting configuration file)
    _database_file_path = config.get('default', 'db')
    if os.path.lexists(_database_file_path):
        os.remove(os.path.join(VAR_DIR, _database_file_path))

    # Remove Researcher certificates and keys
    _clear_files(config, 'certificate', ('private_key', 'public_key'))

    # remove Researcher config file
    _clear_config_file_component(config)


def _clear_files(config: Config, section: str, materials: Tuple[str]):
    """Clears files detailed in a config file, for a given section and a tuple of items
    stored in that section, that correspond to a path pointing to a file.

    If the method has cleared all element of a folder (ie the folder has became empty),
    delete the empty folder.

    Args:
        config: Configuration of the component
        section: Section in config
        materials: Keys in section

    """
    for material in materials:
        _path_to_material = os.path.join(CONFIG_DIR,
                                         config.get(section, material))

        if os.path.lexists(_path_to_material):

            print("[INFO] Removing file ", _path_to_material)
            os.remove(_path_to_material)

        _default_folder = os.path.dirname(_path_to_material)
        if os.path.lexists(_default_folder) and not os.listdir(_default_folder):
            # remove default folder containing certificates (if empty)
            print("[INFO] Removing folder ", _default_folder)
            shutil.rmtree(_default_folder)

def _clear_config_file_component(config: Config):
    """Clears configuration file of a Component (either Node or Researcher)

    Args:
        config: Configuration of the component.
    """
    _component_type = config.get('default', 'component')

    # remove config file
    if config.is_config_existing():
        print("[INFO] Removing file ", config.path)
        os.remove(config.path)

    # TODO: remove temporary file created when using notebook (located in ./var/tmp_xxx)
    print(f"[INFO] {_component_type} with id"
          f"{config.get('default', 'id')} has been cleared")


def clear_experiment_data(exp: 'Experiment'):
    """Clears data relative to an Experiment execution, mainly:
    - `ROOT/experiments/Experiment_xx` folder
    - `ROOT/runs` folder when activating Tensorboard feature

    Args:
        exp: Experiment object used for running experiment
    """
    # removing only big files created by Researcher (for now)
    # remove tensorboard logs (if any)

    print("Stopping gRPC server started by the test function")
    print("Will wait 10 seconds to cancel current RPC requests")

    # Stop GRPC server and remove request object for next experiments

    if not exp._reqs._grpc_server._server._loop.is_closed():

        future = asyncio.run_coroutine_threadsafe(
            exp._reqs._grpc_server._server.stop(10),
            exp._reqs._grpc_server._server._loop
        )

        future.result()

    # Need to remove request
    print("Removing request object")
    from fedbiomed.researcher.requests import Requests
    if Requests in Requests._objects:
        del Requests._objects[Requests]

    tensorboard_folder = os.path.join(ROOT_DIR, TENSORBOARD_FOLDER_NAME)
    tensorboard_files = os.listdir(tensorboard_folder)
    for file in tensorboard_files:
        shutil.rmtree(os.path.join(tensorboard_folder, file))
    print("[INFO] Removing folder content ", tensorboard_folder)

    # remove breakpoints folder created during experimentation from the default folder (if any)
    _exp_dir = os.path.join(VAR_DIR, "experiments")
    current_experimentation_folder = os.path.join(_exp_dir, exp._experimentation_folder)

    print("[INFO] Removing breakpoints", current_experimentation_folder)
    if os.path.isdir(current_experimentation_folder):
        shutil.rmtree(current_experimentation_folder)


def create_component(
    component_type: ComponentType,
    config_name: str,
    config_sections: Dict[str, Dict[str, Any]] = None,
    use_prefix: bool = True
) -> Config:
    """Creates component configuration

    Args:
        component_type: Component type researcher or node
        config_name: name of the config file. Prefix will be added automatically
        config_sections: To overwrite some default configurations in config files.
    Returns:
        config object after prefix added for end to end tests
    """

    if component_type == ComponentType.NODE:
        config = importlib.import_module("fedbiomed.node.config").NodeConfig
    elif component_type == ComponentType.RESEARCHER:
        config = importlib.import_module("fedbiomed.researcher.config").ResearcherConfig

    config_name = f"{CONFIG_PREFIX}{config_name}" if use_prefix else config_name
    config = config(name=config_name, auto_generate=False)

    # If there is already a component created first clear everything and recreate
    # if os.path.isfile(os.path.join(CONFIG_DIR, config_name)):
    #    config.generate()
    #    clear_component_data(config)

    # Need to remove secagg table singleton
    # because it was created when we import from researcher modules
    # + may be re-created during each test
    print("Removing _SecaggTableSingleton object")
    from fedbiomed.common.secagg_manager import _SecaggTableSingleton
    if _SecaggTableSingleton in _SecaggTableSingleton._objects:
        del _SecaggTableSingleton._objects[_SecaggTableSingleton]

    # For now, executing in another process is not mandatory
    # This is provision for future to prevent in config generation event
    # (eg: creating a singleton object) to propagate to this process
    fork_process(config.generate)

    # need to update configuration in parent process
    config.read()


    if config_sections:
        for section, value in config_sections.items():
            if section not in config.sections():
                raise ValueError(f'Section is not in config sections {section}')
            for key, val in value.items():
                config.set(section, key, val)
        # Rewrite after modification
        config.write()
    return config


def create_researcher(
    port: str,
    config_sections: Dict | None = None
) -> Config:
    """Creates researcher component"""

    config_sections = config_sections or {}
    config_sections.update({'server': {'port': port}})

    researcher = create_component(
        ComponentType.RESEARCHER,
        config_name=f"config_researcher_{uuid.uuid4()}.ini",
        config_sections=config_sections,
    )

    os.environ['RESEARCHER_CONFIG_FILE'] = researcher.name

    return researcher

def training_plan_operation(
    config: Config,
    operation: str,
    training_plan_id: str
):
    """Applies approve or reject operation on given config of node

    Args:
        config: Configuration of component, should be node
        operation: One of approve, reject
        training_plan_id: Id of the training plan that the operation will be applied to
    """


    if not operation in ['approve', 'reject']:
        raise ValueError('The argument operation should be one of apprive or reject')


    command = ["node", "--config", config.name, "training-plan",
               operation, "--id", training_plan_id]
    _ = fedbiomed_run(command, wait=True, on_failure=default_on_failure)


def get_data_folder(path):
    """Gets path to save datasets, and creates folder if not existing


    Args:

    """
    ci_data_path = os.environ.get('FEDBIOMED_E2E_DATA_PATH')
    if ci_data_path:
        folder = os.path.join(ci_data_path, path)
    else:
        folder = os.path.join(ROOT_DIR, 'data', path)

    if not os.path.isdir(folder):
        print(f"Data folder for {path} is not existing. Creating folder...")
        os.makedirs(folder)

    return folder

def create_node(port, config_sections:Dict | None = None):
    """Creates node component"""

    c_com = functools.partial(create_component,
        component_type=ComponentType.NODE,
        config_name=f"config_e2e_{uuid.uuid4()}.ini")

    config_sections = config_sections or {}
    config_sections.update({'researcher': {'port': port}})

    return c_com(config_sections=config_sections)


@contextmanager
def create_multiple_nodes(
    port: int,
    num_nodes: int,
    config_sections: Dict | List[Dict] = None
) -> Tuple:
    """Creates multiple node in a context manager"""


    if config_sections:
        if isinstance(config_sections, dict):
            config_sections = [config_sections] * num_nodes
        elif isinstance(config_sections, list) and len(config_sections) != num_nodes:
            raise ValueError(
                f"Number of nodes {num_nodes} is not equal number of config "
                f"sections {len(config_sections)}")
        else:
            raise TypeError(f'Invalid config_sections type {type(config_sections)}')

    # Create nodes
    nodes = []
    for n in range(num_nodes):
        if config_sections:
            nodes.append(create_node(port, config_sections[n]))
        else:
            nodes.append(create_node(port))

    yield tuple(nodes)


    # Clear node data
    for node in nodes:
        clear_node_data(node)



