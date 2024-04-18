"""Contains execution helpers"""

import os
import subprocess
import multiprocessing

from typing import Optional, Callable, List

import psutil
import pytest

# TODO: When it is raised should exit from
# subprocess and parent process
class End2EndErrorExit(SystemExit):
    pass


FEDBIOMED_RUN = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "scripts", "fedbiomed_run")
)

FEDBIOMED_ENVIRONMENT = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "scripts", "fedbiomed_environment")
)


def collect(process, on_failure: Optional[Callable] = None) -> bool:
    """Collects process results. Waits until processes finishes and
    checks returncode

    Args:
        process: A subprocess object
    """

    try:
        returncode = process.wait()
    except Exception as e:
        print(f"Error raised while waiting for the process to finish: {e}")
    else:
        # -9 is for killing the process Ctrl-z or psutil.kill
        if returncode not in [0, -9]:
            if on_failure:
                on_failure(process)
            # Other exceptions are caught by pytest
            pytest.exit(f"Error: Processes failed {process}. "
                         "Please check the outputs.")

    return True

def execute_in_paralel(
    processes: List[subprocess.Popen],
    on_failure: Callable | None = None,
    interrupt_all_on_fail: bool = True
):
    """Execute commands in parallel"""
    def error_callback(err):
        if interrupt_all_on_fail:
            kill_subprocesses(processes)
        print("One of the parallel processes has faild. {err}")

    def collect_result(process):
        collect(process, on_failure)

    with multiprocessing.pool.ThreadPool(100) as pool:
        r = pool.map_async(collect_result, processes, error_callback=error_callback)
        try:
            r.get()
        except Exception as e:
            raise End2EndErrorExit(
                f'Exception raised in one of subprocess. {e}'
            ) from e

def kill_subprocesses(processes):
    """Kills given processes

    Args:
        processes: List of subprocesses to kill
    """
    for p in processes:

        if not psutil.pid_exists(p.pid):
            continue

        parent = psutil.Process(p.pid)
        print("PARENT")
        print(parent.cmdline())
        for child in parent.children(recursive=True):
            print(child.cmdline())
            child.kill()
        parent.kill()



def shell_process(
    command: list,
    activate: str = None,
    wait: bool = False,
    pipe: bool = True,
):
    """Executes shell process

    Args:
        command: List of commands (do not add fedbiomed run it is
            automatically added)
        activate: Name of the component that the conda environment will be activated
            before executing the command
        wait: If true function will block until the command is completed. Otherwise,
            it will return process object that is running in the background.
    """

    if activate:
        if activate not in ["node", "researcher"]:
            ValueError(f"Please select 'node' or 'researcher' not '{activate}'")

        command[:0] = ["source", FEDBIOMED_ENVIRONMENT, activate, ";"]

    pipe_ = True

    if wait:
        pipe_ = False
    elif not pipe:
        pipe_ = False

    print(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen( " ".join(command),
                                shell=True,
                                stdout=subprocess.PIPE if pipe else None,
                                stderr=subprocess.STDOUT if pipe else None,
                                bufsize=1,
                                close_fds=True,
                                universal_newlines=True
        )

    if wait:
        return collect(process)

    return process


def fedbiomed_run(command: list[str], wait: bool = False, pipe: bool = True):
    """Executes given command using fedbiomed_run

    Args:
        command: List of command
        wait: Wait until command is completed (blocking or non-blocking option)
    """
    command.insert(0, FEDBIOMED_RUN)
    return shell_process(command=command, wait=wait, pipe=pipe)
