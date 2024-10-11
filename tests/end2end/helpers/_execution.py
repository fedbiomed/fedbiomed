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
        # 137: Process finished with exit code 137
        # (interrupted by signal 9: SIGKILL) mostly on ubuntu slave
        if returncode not in [0, -9, 137]:
            if on_failure:
                on_failure(process)
            # Other exceptions are caught by pytest
            pytest.exit(f"Error: Processes failed {process}. Args: {process.args} "
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
        print(f"One of the parallel processes has faild. {err}")

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
        kill_process(p)

def kill_process(process):
    """Kills single process"""

    if not psutil.pid_exists(process.pid):
        cmdl = process.cmdline() if hasattr(process, 'cmdline') else process
        print(f"Process is no longer available {cmdl}")
        return

    parent = psutil.Process(process.pid)
    print(f'Checking child process of parent "{parent.cmdline()}"')
    for child in parent.children(recursive=True):
        print(f"Killing child process {child.cmdline()}")
        child.kill()

    try:
        parent.kill()
    except psutil.NoSuchProcess:
        print('Parent process no longer existing after killing child procesess')
    except psutil.ZombieProcess:
        print('Parent process has became zombie process after killing child procesess')


def fork_process(
    command: Callable,
    *args,
    **kwargs,
):
    """Executes forked process

    Args:
        command: Command to execute in forked process
        *args: optional args
        **kwargs: optional kwargs
    """
    context = multiprocessing.get_context('spawn')
    p = context.Process(target=command, args=args, kwargs=kwargs)
    p.run()

    if isinstance(p.exitcode, int) and p.exitcode != 0:
        # Other exceptions are caught by pytest
        pytest.exit(f"Error: Forked process failed {command}. Args: {args} {kwargs} "
                    "Please check the outputs.")


def shell_process(
    command: list,
    activate: str = None,
    wait: bool = False,
    pipe: bool = True,
    on_failure: Callable | None = None
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
        if not activate in ["node", "researcher"]:
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
                                stdout=subprocess.PIPE if pipe_ else None,
                                stderr=subprocess.STDOUT if pipe_ else None,
                                bufsize=1,
                                close_fds=True,
                                universal_newlines=True
        )

    if wait:
        return collect(process, on_failure)

    return process


def fedbiomed_run(
    command: list[str],
    wait: bool = False,
    pipe: bool = True,
    on_failure: Callable | None = None
) -> subprocess.Popen:
    """Executes given command using fedbiomed_run

    Args:
        command: List of command
        wait: Wait until command is completed (blocking or non-blocking option)
    """
    command.insert(0, FEDBIOMED_RUN)
    return shell_process(command=command, wait=wait, pipe=pipe, on_failure=on_failure)


