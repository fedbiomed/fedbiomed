
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
        process.wait()
    except Exception as e:
        print(f"Error raised! {e}")

    if process.returncode != 0:
        raise End2EndError("Error")


def default_on_exit(process: subprocess.Popen):
    """Default function to execute when the process is on exit"""

    if process.returncode != 0:
        raise End2EndError(f"Processes has stopped with error. {process.stderr.readline()}")

    print("Process has finshed without error")



def execute_in_paralel(processes: subprocess.Popen, on_exit = default_on_exit):
    """Execute commands in parallel"""

    timeout = 0.1
    while processes:
        for p in processes:
            if p.poll() is not None:
                print(p.stdout.read(), end="")
                p.stdout.close()

                # Execute on exit
                on_exit(p)

                # Remove process from list
                processes.remove(p)

        rlist = select.select([p.stdout for p in processes], [], [], timeout)[0]
        rlist_err = select.select([p.stderr for p in processes], [], [], timeout)[0]

        for f  in rlist:
            print(f.readline(), end='')

        for f in rlist_err:
            print(f.readline(), end='')


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
                                stderr=subprocess.PIPE,
                                bufsize=1,
                                close_fds=True,
                                universal_newlines=True
                                )


    return process
