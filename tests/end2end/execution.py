
import sys
import os
import select
import subprocess

from constants import End2EndError

FEDBIOMED_RUN = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "scripts", "fedbiomed_run")
)

def collect(process):
    """Collects process results"""

    try:
        output, error = process.communicate()
    except Exception as e:
        print(f"Error raised! {e}")


    print(output)
    if process.returncode != 0:
        raise End2EndError(f"Error: {error}")


    return True

def default_on_exit(process: subprocess.Popen):
    """Default function to execute when the process is on exit"""

    if process.returncode != 0:
        raise End2EndError(f"Processes has stopped with error. {process.stderr.readline()}")

    raise Exception("Processes finished")
    print("Process is finshed!")



def execute_in_paralel(processes: subprocess.Popen, on_exit = default_on_exit):
    """Execute commands in parallel"""

    while processes:
        for p in processes:
            line = p.stdout.readline().strip()
            print(line)
            if p.poll() != None:
                print("Process finished")
                print(f"Return code {p.returncode}")
                processes.remove(p)



def shell_process(command: list):
    """Executes shell process

    Args:
        command: List of commands (do not add fedbiomed run it is
            automatically added)
    """

    command.insert(0, FEDBIOMED_RUN)
    process = subprocess.Popen( " ".join(command),
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                bufsize=1,
                                close_fds=True,
                                universal_newlines=True
                                )


    return process
