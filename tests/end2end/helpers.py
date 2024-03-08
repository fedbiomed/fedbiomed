import importlib
import shutil
import tempfile
import json
import os
import threading
import multiprocessing
import psutil

from execution import shell_process, fedbiomed_run, collect, execute_in_paralel
from constants import CONFIG_PREFIX, End2EndError


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.config import Config
from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR


def create_component(
    component_type: ComponentType,
    config_name:str
) -> Config:
    """Creates component configuration

    Args:
        component_type: Component type researcher or node
        config_name: name of the config file. Prefix will be added automatically

    Returns:
        config object after prefix added for end to end tests
    """

    if component_type == ComponentType.NODE:
        config = importlib.import_module("fedbiomed.node.config").NodeConfig
    elif component_type == ComponentType.RESEARCHER:
        config = importlib.import_module("fedbiomed.researcher.config").ResearcherConfig

    config_name = f"{CONFIG_PREFIX}{config_name}"

    config = config(name=config_name, auto_generate=False)

    config.generate()


    return config


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
    process = fedbiomed_run(command, wait=True)

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
    t = threading.Thread(target=execute_in_paralel, args=(processes,))
    t.start()

    return processes, t

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

    print("Pytohn script execution")
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

        # load node 's environ
        # environ = importlib.import_module("fedbiomed.node.environ").environ
        # print("ENVIRON", environ["RESEARCHERS"])
        # print("NODE_ID", environ["NODE_ID"])

        node_id = config.get('default', 'id')
        # remove node's state
        _node_state_dir = os.path.join(VAR_DIR, "node_state_%s" % node_id)

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
        # remove config file
        if  config.is_config_existing():
            print("[INFO] Removing file ", config.path)
            os.remove(config.path)
        print(f"[INFO] {_component_type} has been cleared")


    elif _component_type == ComponentType.RESEARCHER.name:
        # TODO: complete below for Researcher
        pass
