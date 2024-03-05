
import importlib
import tempfile
import json
import os
import threading
import multiprocessing

from execution import shell_process, collect, execute_in_paralel
from constants import CONFIG_PREFIX

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.config import Config

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
) -> str:
    """Adds given dataset using given configuration of the node"""

    tempdir_ = tempfile.TemporaryDirectory()
    d_file = os.path.join(tempdir_.name, "dataset.json")
    with open(d_file, "w", encoding="UTF-8") as file:
        json.dump(dataset, file)

    command = ["node", "--config", config.name, "dataset", "add", "--file", d_file]
    process = shell_process(command)
    collect(process)

    tempdir_.cleanup()

    return True


def _start_nodes(
        configs: list[Config],
) -> bool:
    """Starts given nodes"""

    processes = []
    for c in configs:
        processes.append(shell_process(["node", "--config", c.name, "start"]))

    execute_in_paralel(processes)


def start_nodes(
    configs: list[Config]
) -> multiprocessing.Process:
    """Starts the nodes by given list of configs

    Args:
        configs: List of node config objects
    """

    p = multiprocessing.Process(target=_start_nodes, args=(configs, ))
    p.deamon = True
    p.start()

    return p


def execute_python(file: str):
    """Executes given python file in a process"""
    return file


def execute_ipython(file: str):
    """Executes given ipython file in a process"""

    return file


def clear_component_data(config: Config):
    """Clears component related file"""
    pass
