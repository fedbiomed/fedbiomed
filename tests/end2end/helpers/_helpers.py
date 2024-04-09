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
import threading
import multiprocessing

from typing import Dict, List, Any

import psutil

from fedbiomed.common.constants import TENSORBOARD_FOLDER_NAME, ComponentType
from fedbiomed.common.config import Config
from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR
from fedbiomed.researcher.experiment import Experiment

from ._execution import (
    shell_process,
    fedbiomed_run,
    collect_output_in_parallel
)
from .constants import CONFIG_PREFIX, FEDBIOMED_SCRIPTS, End2EndError


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
    _ = fedbiomed_run(command, wait=True)
    tempdir_.cleanup()

    return True


def start_nodes(
    configs: list[Config]
) -> multiprocessing.Process:
    """Starts the nodes by given list of configs

    Args:
        configs: List of node config objects
    """

    processes = []
    for c in configs:
        processes.append(fedbiomed_run(["node", "--config", c.name, "start"]))

     # Listen outputs in parallel
    t = threading.Thread(target=collect_output_in_parallel, args=(processes,))
    t.start()

    return processes, t


def configure_secagg():
    """Configures secure aggregation environment"""

    # Darwin does long installation therefore if shamir protocol is already
    # complied do not do other compilation.
    if platform.system() == 'Darwin':
        MP_SPDZ_BASEDIR = os.path.join(FEDBIOMED_SCRIPTS, '..', 'modules', 'MP-SPDZ')
        if os.path.isfile(os.path.join(MP_SPDZ_BASEDIR, 'shamir-party.x')):
            print("MP-SDPZ is already configured")
            return

    script = os.path.join(FEDBIOMED_SCRIPTS, 'fedbiomed_configure_secagg')
    _ = shell_process(
        [f"HOMEBREW_NO_INSTALL_FROM_API=1", script, 'node'],
        wait=True
    )


def secagg_certificate_registration():
    """Registers certificates of all components whose configs are available"""

    return fedbiomed_run(['certificate-dev-setup'], wait=True)


def kill_subprocesses(processes):
    """Kills given processes

    Args:
        processes: List of subprocesses to kill
    """
    for p in processes:
        parent = psutil.Process(p.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()


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
        wait=True
    )

def execute_ipython(file: str, activate: str):
    """Executes given ipython file in a process"""

    return shell_process(
        command=["ipython", "-c", f'"%run {file}"'],
        activate=activate,
        wait=True
    )


def clear_component_data(config: Config):
    """Clears component related file"""

    # extract Component Type from config file

    _component_type = config.get('default', 'component')

    if _component_type == ComponentType.NODE.name:

       clear_node_data(config)

    elif _component_type == ComponentType.RESEARCHER.name:

        # remove Researcher database
        # researcher_db_file = os.path.join()

        # remove Researcher config file
        pass


def clear_node_data(config: Config):
    """Clears data relative to Node, such as configuration file, database,
    node state, mpspdz material

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

    # remove grpc certificate
    for section in config.sections() :
        if section.startswith("researcher"):
            # _certificate_file = environ["RESEARCHERS"][0]['certificate']
            # if _certificate_file:
            #     os.remove(os.path.join(CONFIG_DIR, _certificate_file))

            # TODO: find a way or modify environ in order to delete GRPC certificate
            pass

    # remove node's mpspdz material
    _mpspdz_material_files = ('private_key', 'public_key')
    for mpspdz_file in _mpspdz_material_files:
        mpspdz_material = config.get('mpspdz', mpspdz_file,)
        _material_to_remove = os.path.join(CONFIG_DIR, mpspdz_material)
        _material_to_remove_folder = os.path.dirname(_material_to_remove)
        if not os.path.lexists(_material_to_remove_folder):
            continue
        print("[INFO] Removing folder ", _material_to_remove_folder)
        shutil.rmtree(_material_to_remove_folder)  # remove the whole folder of cert

    # remove database
    # FIXME: below we assume database is in the `VAR_DIR` folder
    _database_file_path = config.get('default', 'db')

    os.remove(os.path.join(VAR_DIR, _database_file_path))
    # remove Node's config file
    _clear_config_file_component(config)


def clear_researcher_data(config: Config):
    """Clears data relative to Researcher"""
    # TODO: continue implementing this method
    raise NotImplementedError
    # remove Researcher config file
    _clear_config_file_component(config)


def _clear_config_file_component(config: Config):
    """Clears configuration file of a Component (either Node or Researcher)
    """
    _component_type = config.get('default', 'component')
    # remove config file
    if config.is_config_existing():
        print("[INFO] Removing file ", config.path)
        os.remove(config.path)

    # TODO: remove temporary file created when using
    # notebook (located in ./var/tmp_xxx)
    print(f"[INFO] {_component_type} with id"
          f"{config.get('default', 'id')} has been cleared")


def clear_experiment_data(exp: Experiment):
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
    # _nb_exp_folders = len(os.listdir(_exp_folder))
    # current_experimentation_folder = "Experiment_" + str("{:04d}".format(_nb_exp_folders - 1))
    # current_experimentation_folder = os.path.join(_exp_folder, current_experimentation_folder)
    current_experimentation_folder = os.path.join(_exp_dir, exp._experimentation_folder)

    print("[INFO] Removing breakpoints", current_experimentation_folder)
    shutil.rmtree(current_experimentation_folder)


def create_component(
    component_type: ComponentType,
    config_name: str,
    config_sections: Dict[str, Dict[str, Any]] = None
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

    config_name = f"{CONFIG_PREFIX}{config_name}"
    config = config(name=config_name, auto_generate=False)

    # If there is already a component created first clear everything and recreate
    if os.path.isfile(os.path.join(CONFIG_DIR, config_name)):
        config.generate()
        clear_component_data(config)

    config.generate()

    if config_sections:
        for section, value in config_sections.items():
            if not section in config.sections():
                raise ValueError(f'Section is not in config sections {section}')
            for key, val in value.items():
                config.set(section, key, val)
        # Rewrite after modification
        config.write()
    return config
