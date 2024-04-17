"""Contains execution helpers"""

import os
import subprocess


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



def collect(process) -> bool:
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
        if returncode != 0:
            raise End2EndErrorExit(f"Error: Processes failed. Please check the outputs")

    return True

def default_on_exit(process: subprocess.Popen):
    """Default function to execute when the process is on exit"""

    if process.returncode not in [0, -9]:
        raise End2EndErrorExit(f"Processes has stopped with error. {process.stdout.readline()}")

    print("Process is finished!")


def collect_output_in_parallel(
    processes: subprocess.Popen,
    on_exit = default_on_exit
):
    """Execute commands in parallel"""

    while processes:
        for p in processes:
            line = p.stdout.readline().strip()
            print(line)
            if p.poll() != None:
                processes.remove(p)
                on_exit(p)

def shell_process(
    command: list,
    activate: str = None,
    wait: bool = False,
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


    print(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen( " ".join(command),
                                shell=True,
                                stdout=subprocess.PIPE if not wait else None,
                                stderr=subprocess.STDOUT if not wait else None,
                                bufsize=1,
                                close_fds=True,
                                universal_newlines=True
        )

    if wait:
        return collect(process)

    return process


def fedbiomed_run(command: list[str], wait: bool = False):
    """Executes given command using fedbiomed_run

    Args:
        command: List of command
        wait: Wait until command is completed (blocking or non-blocking option)
    """
    command.insert(0, FEDBIOMED_RUN)
    return shell_process(command=command, wait=wait)
