
import importlib
import shutil
import tempfile
import json
import os
import threading
import multiprocessing
import subprocess
import psutil

from execution import shell_process, collect, execute_in_paralel, FEDBIOMED_RUN
from constants import CONFIG_PREFIX

from fedbiomed.common.constants import TENSORBOARD_FOLDER_NAME, ComponentType
from fedbiomed.common.config import Config
from fedbiomed.common.utils import ROOT_DIR, CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR
from fedbiomed.researcher.experiment import Experiment


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
    # command.insert(0, FEDBIOMED_RUN)
    # subprocess.call(command)
    process = shell_process(command)
    collect(process)

    tempdir_.cleanup()

    return True



def _start_nodes(
        configs: list[Config],
) -> bool:
    """Starts given nodes"""

    print("Starting nodes")
    processes = []
    for c in configs:
        print(f"Starting node start process for config {c.name}")
        processes.append(shell_process(["node", "--config", c.name, "start"]))
        print(f"Process created for {c.name}")

    print("Executin in paralel!")
    execute_in_paralel(processes)


def start_nodes(
    configs: list[Config]
) -> multiprocessing.Process:
    """Starts the nodes by given list of configs

    Args:
        configs: List of node config objects
    """

    processes = []
    for c in configs:
        processes.append(shell_process(["node", "--config", c.name, "start"]))


     # Listen outputs in parallel
    t = threading.Thread(target=execute_in_paralel, args=(processes,))
    t.start()


    return processes, t

def kill_subprocesses(processes):
    """Kills given processes"""
    for p in processes:

        print(f"Killing process: {p.pid} and it childs")
        parent = psutil.Process(p.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

def execute_python(file: str):
    """Executes given python file in a process"""
    return file


def execute_ipython(file: str):
    """Executes given ipython file in a process"""

    return file


def clear_component_data(config: Config):
    """Clears component related file"""

    # extract Component Type from config file

    _component_type = config.get('default', 'component')

    if _component_type == ComponentType.NODE.name:

        clear_node_data(config)
        
        
    elif _component_type == ComponentType.RESEARCHER.name:
        
        # remove Researcher database
        #researcher_db_file = os.path.join()

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

    # TODO: remove temporary file created when using notebook (located in ./var/tmp_xxx)
    print(f"[INFO] {_component_type} with id {config.get('default', 'id')} has been cleared")

def clear_experiment_data(exp: Experiment):
    """Clears data relative to an Experiment execution, mainly:
    - `ROOT/experiments/Experiment_xx` folder
    - `ROOT/runs` folder when activating Tensorboard feature

    Args:
        exp: Experiment object used for running experiment
    """
    # removing only big files created by Researcher (for now) 
    # remove tensorboard logs (if any)
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
