"""
Helper methods for end2end tests
"""

import asyncio
import importlib
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from fedbiomed.researcher.federated_workflows import Experiment

from fedbiomed.common.config import Config
from fedbiomed.common.constants import ComponentType

from ._execution import (
    execute_in_paralel,
    fedbiomed_run,
    kill_subprocesses,
    shell_process,
)
from .constants import CONFIG_PREFIX, End2EndError

# Cached outside the per-module workspace so each dataset is fetched once.
DEFAULT_DATA_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "fedbiomed", "e2e-data"
)

# Only one researcher can exist per process, so only one federation can.
_federation: "Federation | None" = None


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


def add_dataset_to_node(config: Config, dataset: dict) -> bool:
    """Adds given dataset using given configuration of the node"""

    tempdir_ = tempfile.TemporaryDirectory()
    d_file = os.path.join(tempdir_.name, "dataset.json")
    with open(d_file, "w", encoding="UTF-8") as file:
        json.dump(dataset, file)

    command = ["node", "--path", config.root, "dataset", "add", "--file", d_file]
    _ = fedbiomed_run(command, wait=True, on_failure=default_on_failure)
    tempdir_.cleanup()

    return True


def default_on_failure(process: subprocess.Popen):
    """Default function to execute when the process is on exit"""
    print(f"On failure callback: Process has failed!, {process}")
    raise End2EndError(f"Processes has failed! command: {process.args}")


def start_nodes(
    configs: list[Config],
    interrupt_all_on_fail: bool = True,
    on_failure: Callable = default_on_failure,
) -> Tuple[multiprocessing.Process, PytestThread]:
    """Starts the nodes by given list of configs

    Args:
        configs: List of node config objects
    """

    processes = []
    for c in configs:
        # Keep it for debugging purposes
        if "fail_my_component" in c.root:
            processes.append(
                fedbiomed_run(["node", "--path", c.root, "unkown-commnad"], pipe=False)
            )
        else:
            processes.append(
                fedbiomed_run(["node", "--path", c.root, "start"], pipe=False)
            )

    t = PytestThread(
        target=execute_in_paralel,
        kwargs={
            "processes": processes,
            "interrupt_all_on_fail": interrupt_all_on_fail,
            "on_failure": on_failure,
        },
    )

    t.daemon = True
    t.start()

    return processes, t


def execute_script(file: str, activate: str = "researcher"):
    """Executes given scripts"""

    if not os.path.isfile(file):
        raise End2EndError("file is not existing")

    if file.endswith(".py"):
        return execute_python(file, activate)

    if file.endswith(".ipynb"):
        return execute_ipython(file, activate)

    raise End2EndError("Unsupported file type. Please use .py or .ipynb")


def execute_python(file: str, activate: str):
    """Executes given python file in a process"""

    return shell_process(
        command=["python", f"{file}"], wait=True, on_failure=default_on_failure
    )


def execute_ipython(file: str, activate: str):
    """Executes given ipython file in a process"""

    return shell_process(
        command=["ipython", "-c", f'"%run {file}"'],
        wait=True,
        on_failure=default_on_failure,
    )


def clear_component_data(config: Config):
    """Clears component related file"""

    # extract Component Type from config file

    shutil.rmtree(config.root)


def stop_grpc_server(grpc_server):
    """Stops the researcher gRPC server and joins its thread."""
    # A closed loop means the server thread is already winding down, so only the
    # join below can tell whether it actually finished.
    if not grpc_server._server._loop.is_closed():
        future = asyncio.run_coroutine_threadsafe(
            grpc_server._server.stop(10),
            grpc_server._server._loop,
        )
        print("##### FBM: Waiting for server to stop, timeout after 10 seconds")
        try:
            future.result(10)
        except Exception as e:
            print(
                "#### FBM: Exception has raised while stopping gRPC server."
                f"Timeout: 10, Error: {e}"
            )
            try:
                grpc_server._server._loop.stop()
            except Exception as e:
                print(f"#### FBM: Error while closing loop: {e}")

    grpc_server._thread.join(timeout=10)
    if grpc_server._thread.is_alive():
        raise End2EndError(
            "Researcher gRPC server thread is still alive after the join timeout "
            f"on port {grpc_server._port}"
        )

    print("##### FBM: Researcher server has stopped")


def stop_researcher_server():
    """Stops the researcher gRPC server and drops the `Requests` singleton.

    Serves both the end of an experiment and the module teardown safety net. The
    singleton is dropped even when the server refuses to stop, so a later test
    still starts from a clean state while the failure is reported.
    """
    from fedbiomed.researcher.requests import Requests

    requests = Requests._objects.get(Requests)
    if requests is None:
        return

    try:
        stop_grpc_server(requests._grpc_server)
    finally:
        Requests._objects.pop(Requests, None)


def clear_experiment_data(exp: "Experiment"):
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

    # `exp._reqs` is the `Requests` singleton this stops and then drops, so the
    # next experiment of the module starts its own server.
    stop_researcher_server()

    # tensorboard_folder = os.path.join(config.root, TENSORBOARD_FOLDER_NAME)
    # tensorboard_files = os.listdir(tensorboard_folder)
    # for file in tensorboard_files:
    #    shutil.rmtree(os.path.join(tensorboard_folder, file))
    # print("[INFO] Removing folder content ", tensorboard_folder)

    # remove breakpoints folder created during experimentation from the default folder (if any)
    # _exp_dir = os.path.join(config.root, VAR_FOLDER_NAME, "experiments")
    # current_experimentation_folder = os.path.join(_exp_dir, exp._experimentation_folder)

    # print("[INFO] Removing breakpoints", current_experimentation_folder)
    # if os.path.isdir(current_experimentation_folder):
    #    shutil.rmtree(current_experimentation_folder)


def create_component(
    component_type: ComponentType,
    directory: str,
    component_name: str,
    config_sections: Dict[str, Dict[str, Any]] = None,
    use_prefix: bool = True,
) -> Config:
    """Creates component configuration

    Args:
        component_type: Component type researcher or node
        component_name: Name of the component directory. Prefix will be added automatically.
        config_sections: To overwrite some default configurations in config files.
    Returns:
        config object after prefix added for end to end tests
    """

    if component_type == ComponentType.NODE:
        comp = importlib.import_module("fedbiomed.node.config").node_component
    elif component_type == ComponentType.RESEARCHER:
        comp = importlib.import_module(
            "fedbiomed.researcher.config"
        ).researcher_component
    else:
        raise ValueError(f"Urecognized component type {component_type}")

    component_name = (
        f"{CONFIG_PREFIX}{component_name}" if use_prefix else component_name
    )
    root = os.path.join(directory, component_name)
    config = comp.initiate(root=root)

    # Need to remove secagg table singleton
    # because it was created when we import from researcher modules
    # + may be re-created during each test
    print("Removing _SecaggTableSingleton object")
    from fedbiomed.common.secagg_manager import _SecaggTableSingleton

    if _SecaggTableSingleton in _SecaggTableSingleton._objects:
        del _SecaggTableSingleton._objects[_SecaggTableSingleton]

    # need to update configuration in parent process
    config.read()

    if config_sections:
        for section, value in config_sections.items():
            if section not in config.sections():
                raise ValueError(f"Section is not in config sections {section}")
            for key, val in value.items():
                config.set(section, key, val)
        # Rewrite after modification
        config.write()
    return config


def create_researcher(
    directory: str, port: str, config_sections: Dict | None = None
) -> Config:
    """Creates the researcher component files under the given directory"""

    config_sections = config_sections or {}
    config_sections.update({"server": {"port": port}})

    researcher = create_component(
        ComponentType.RESEARCHER,
        directory=directory,
        component_name=f"config_researcher_{uuid.uuid4()}.ini",
        config_sections=config_sections,
    )
    os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = researcher.root
    from fedbiomed.researcher.config import config

    config.load(root=researcher.root)

    return researcher


def training_plan_operation(config: Config, operation: str, training_plan_id: str):
    """Applies approve or reject operation on given config of node

    Args:
        config: Configuration of component, should be node
        operation: One of approve, reject
        training_plan_id: Id of the training plan that the operation will be applied to
    """

    if operation not in ["approve", "reject"]:
        raise ValueError("The argument operation should be one of approve or reject")

    command = [
        "node",
        "--path",
        config.root,
        "training-plan",
        operation,
        "--id",
        training_plan_id,
    ]
    _ = fedbiomed_run(command, wait=True, on_failure=default_on_failure)


def get_data_folder(path):
    """Returns the path for storing datasets, creating the folder if it does not exist.

    The data root outlives the module workspace: downloaders skip work when the
    files are already there, so each dataset is fetched once. Clear the root to
    recover from a truncated archive.

    Args:
        path: Relative sub-path appended to the data root.
    """
    root = os.environ.get("FEDBIOMED_E2E_DATA_PATH") or DEFAULT_DATA_CACHE
    folder = os.path.join(root, path)

    if not os.path.isdir(folder):
        print(f"Data folder for {path} is not existing. Creating folder...")
        os.makedirs(folder, exist_ok=True)

    return folder


def create_node(
    directory: str, port: str, config_sections: Dict | None = None
) -> Config:
    """Creates a node component's files under the given directory"""

    config_sections = config_sections or {}
    config_sections.update({"researcher": {"port": port}})

    return create_component(
        ComponentType.NODE,
        directory=directory,
        component_name=f"config_e2e_{uuid.uuid4()}.ini",
        config_sections=config_sections,
    )


class Federation:
    """The researcher and the nodes of one test module.

    A process holds one researcher only: `Requests` is a singleton,
    `FBM_RESEARCHER_COMPONENT_ROOT` is process-wide and the researcher config is
    module-global. Tests build one federation and add nodes to it, including the
    extra nodes an individual test needs.

    Owns the components it creates and the processes it starts, so tests never
    handle teardown order.
    """

    def __init__(self, directory: str, port: str) -> None:
        self._directory = directory
        self._port = port
        self._groups: List[Tuple[List, PytestThread]] = []
        self.researcher = create_researcher(directory, port)

    @contextmanager
    def nodes(self, count: int = 1, config_sections: Dict | None = None) -> Tuple:
        """Adds nodes to the federation for the duration of the block.

        Whatever the block started is stopped on exit and the components
        removed, so test-scoped nodes cannot outlive their test.
        """
        nodes = tuple(
            create_node(self._directory, self._port, config_sections)
            for _ in range(count)
        )
        started = len(self._groups)

        try:
            yield nodes
        except BaseException:
            # The processes may still hold these directories. The workspace
            # removes them, so cleaning here would only risk replacing the real
            # failure with a cleanup error.
            print("Deferring node cleanup after an earlier failure.")
            del self._groups[started:]
            raise
        else:
            self._stop(started)
            for node in nodes:
                clear_component_data(node)

    def start(self, nodes: Tuple[Config, ...]) -> None:
        """Starts the given nodes, keeping their supervisor until they stop."""
        processes, thread = start_nodes(list(nodes))
        self._groups.append((processes, thread))

    def _stop(self, from_index: int = 0) -> None:
        """Stops every group started from the given index onwards.

        Joining is what surfaces a node that died on its own, and it re-raises,
        so everything is killed first to leave no process behind.
        """
        groups = self._groups[from_index:]
        del self._groups[from_index:]

        for processes, _ in groups:
            kill_subprocesses(processes)

        for _, thread in groups:
            thread.join()

    def close(self) -> None:
        """Stops whatever is still running and removes the researcher.

        The researcher goes even when stopping raises, so a reported node death
        leaves nothing behind.
        """
        try:
            self._stop()
        finally:
            clear_component_data(self.researcher)


@contextmanager
def create_federation(directory: str, port: str) -> Federation:
    """Creates the federation a test module runs against."""
    global _federation

    if _federation is not None:
        raise End2EndError(
            "A federation already exists for this process. Only one researcher "
            "can run at a time, so extra nodes must be added to it with "
            "`Federation.nodes` instead of creating a second federation."
        )

    _federation = Federation(directory, port)
    try:
        yield _federation
    finally:
        try:
            _federation.close()
        finally:
            _federation = None
