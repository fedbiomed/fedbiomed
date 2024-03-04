from execution import (
    shell_process,
    collect,
    execute_in_paralel,
)

from constants import CONFIG_PREFIX


def execute_add_mnist():
    """Executes --- """
    command = ["node", "--config", f"{CONFIG_PREFIX}n1.ini", "dataset", "add", "--mnist", "./data"]

    process = shell_process(command)

    try:
        collect(process)
    except Exception as e:
        print("Failed:", e)


# execute_add_mnist()

def start_nodes():

    command_1 = ["node", "--config", f"{CONFIG_PREFIX}n1.ini", "start"]
    command_2 = ["node", "--config123", f"{CONFIG_PREFIX}n2.ini", "start"]

    p1 = shell_process(command_1)
    p2 = shell_process(command_2)

    #p1.wait()
    #p2.wait()
    execute_in_paralel([p1, p2])

# start_nodes()

def start_node():

    command_1 = ["node", "--config", f"{CONFIG_PREFIX}n1.ini", "start"]

    process = shell_process(command_1)
    process.wait()

start_nodes()
