# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Command line user interface for the node component
"""

import json
import os
import signal
import sys
import time
from multiprocessing import Process
from typing import Union
from types import FrameType
import readline

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.environ import environ
from fedbiomed.node.node import Node
from fedbiomed.common.logger import logger
from fedbiomed.common.cli import CommonCLI
from fedbiomed.node.cli_utils import dataset_manager, add_database, delete_database, delete_all_database, \
    tp_security_manager, register_training_plan, update_training_plan, approve_training_plan, reject_training_plan, \
    delete_training_plan, view_training_plan

#
# print(pyfiglet.Figlet("doom").renderText(' fedbiomed node'))
#
__intro__ = """

   __         _ _     _                          _                   _
  / _|       | | |   (_)                        | |                 | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |  _ __   ___   __| | ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _ \ / _` |/ _ \\
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | | | | (_) | (_| |  __/
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_| |_| |_|\___/ \__,_|\___|


"""

# this may be changed on command line or in the config_node.ini
logger.setLevel("DEBUG")

readline.parse_and_bind("tab: complete")


def node_signal_handler(signum: int, frame: Union[FrameType, None]):
    """Signal handler that terminates the process.

    Args:
        signum: Signal number received.
        frame: Frame object received. Currently unused

    Raises:
       SystemExit: Always raised.
    """

    # get the (running) Node object
    global node

    if node:
        node.send_error(ErrorNumbers.FB312)
    else:
        logger.error("Cannot send error message to researcher (node not initialized yet)")
    logger.critical("Node stopped in signal_handler, probably by user decision (Ctrl C)")
    time.sleep(1)
    sys.exit(signum)


def manage_node(node_args: Union[dict, None] = None):
    """Runs the node component and blocks until the node terminates.

    Intended to be launched by the node in a separate process/thread.

    Instantiates `Node` and `DatasetManager` object, start exchaning
    messages with the researcher via the `Node`, passes control to the `Node`.

    Args:
        node_args: command line arguments for node.
            See `Round()` for details.
    """

    global node

    try:
        signal.signal(signal.SIGTERM, node_signal_handler)

        logger.info('Launching node...')

        # Register default training plans and update hashes
        if environ["TRAINING_PLAN_APPROVAL"]:
            # This methods updates hashes if hashing algorithm has changed
            tp_security_manager.check_hashes_for_registered_training_plans()
            if environ["ALLOW_DEFAULT_TRAINING_PLANS"]:
                logger.info('Loading default training plans')
                tp_security_manager.register_update_default_training_plans()
        else:
            logger.warning('Training plan approval for train request is not activated. ' +
                           'This might cause security problems. Please, consider to enable training plan approval.')

        logger.info('Starting communication channel with network')
        node = Node(dataset_manager=dataset_manager,
                    tp_security_manager=tp_security_manager,
                    node_args=node_args)
        node.start_messaging(block=False)

        logger.info('Starting task manager')
        node.task_manager()  # handling training tasks in queue

    except FedbiomedError:
        logger.critical("Node stopped.")
        # we may add extra information for the user depending on the error

    except Exception as e:
        # must send info to the researcher (no mqqt should be handled by the previous FedbiomedError)
        node.send_error(ErrorNumbers.FB300, extra_msg="Error = " + str(e))
        logger.critical("Node stopped.")

    finally:
        # this is triggered by the signal.SIGTERM handler SystemExit(0)
        #
        # cleaning staff should be done here
        pass

    # finally:
    #     # must send info to the researcher (as critical ?)
    #     logger.critical("(CRIT)Node stopped, probably by user decision (Ctrl C)")
    #     time.sleep(1)
    #     logger.exception("Reason:")
    #     time.sleep(1)


def launch_node(node_args: Union[dict, None] = None):
    """Launches a node in a separate process.

    Process ends when user triggers a KeyboardInterrupt exception (CTRL+C).

    Args:
        node_args: Command line arguments for node
            See `Round()` for details.
    """

    p = Process(target=manage_node, name='node-' + environ['NODE_ID'], args=(node_args,))
    p.daemon = True
    p.start()

    logger.info("Node started as process with pid = " + str(p.pid))
    try:
        print('To stop press Ctrl + C.')
        p.join()
    except KeyboardInterrupt:
        p.terminate()

        # give time to the node to send a MQTT message
        time.sleep(1)
        while p.is_alive():
            logger.info("Terminating process id =" + str(p.pid))
            time.sleep(1)

        # (above) p.exitcode returns None if not finished yet
        logger.info('Exited with code ' + str(p.exitcode))

        sys.exit(0)


def launch_cli():
    """Parses command line input for the node component and launches node accordingly.
    """

    cli = CommonCLI()
    cli.set_environ(environ=environ)
    cli.initialize_certificate_parser()
    cli.initialize_create_configuration()

    # Register description for CLI
    cli.description = f'{__intro__}:A CLI app for fedbiomed researchers.'

    cli.parser.add_argument('-a', '--add',
                            help='Add and configure local dataset (interactive)',
                            action='store_true')
    cli.parser.add_argument('-am', '--add-mnist',
                            help='Add MNIST local dataset (non-interactive)',
                            type=str, nargs='?', const='', metavar='path_mnist',
                            action='store')
    # this option provides a json file describing the data to add
    cli.parser.add_argument('-adff', '--add-dataset-from-file',
                            help='Add a local dataset described by json file (non-interactive)',
                            type=str,
                            action='store')
    cli.parser.add_argument('-d', '--delete',
                            help='Delete existing local dataset (interactive)',
                            action='store_true')
    cli.parser.add_argument('-da', '--delete-all',
                            help='Delete all existing local datasets (non interactive)',
                            action='store_true')
    cli.parser.add_argument('-dm', '--delete-mnist',
                            help='Delete existing MNIST local dataset (non-interactive)',
                            action='store_true')
    cli.parser.add_argument('-l', '--list',
                            help='List my shared_data',
                            action='store_true')
    cli.parser.add_argument('-s', '--start-node',
                            help='Start fedbiomed node.',
                            action='store_true')
    cli.parser.add_argument('-rtp', '--register-training-plan',
                            help='Register and approve a training plan from a local file.',
                            action='store_true')
    cli.parser.add_argument('-atp', '--approve-training-plan',
                            help='Approve a training plan (requested, default or registered)',
                            action='store_true')
    cli.parser.add_argument('-rjtp', '--reject-training-plan',
                            help='Reject a training plan (requested, default or registered)',
                            action='store_true')
    cli.parser.add_argument('-utp', '--update-training-plan',
                            help='Update training plan file (for a training plan registered from a local file)',
                            action='store_true')
    cli.parser.add_argument('-dtp', '--delete-training-plan',
                            help='Delete a training plan from database (not for default training plans)',
                            action='store_true')
    cli.parser.add_argument('-ltps', '--list-training-plans',
                            help='List all training plans (requested, default or registered)',
                            action='store_true')
    cli.parser.add_argument('-vtp', '--view-training-plan',
                            help='View a training plan source code (requested, default or registered)',
                            action='store_true')
    cli.parser.add_argument('-g', '--gpu',
                            help='Use of a GPU device, if any available (default: dont use GPU)',
                            action='store_true')
    cli.parser.add_argument('-gn', '--gpu-num',
                            help='Use GPU device with the specified number instead of default device, if available',
                            type=int,
                            action='store')
    cli.parser.add_argument('-go', '--gpu-only',
                            help='Force use of a GPU device, if any available, even if researcher doesnt ' +
                                 'request it (default: dont use GPU)',
                            action='store_true')

    print(__intro__)
    print('\t- 🆔 Your node ID:', environ['NODE_ID'], '\n')

    # Parse CLI arguments after the arguments are ready
    cli.parse_args()

    if cli.arguments.add:
        add_database()
    elif cli.arguments.add_mnist is not None:
        add_database(interactive=False, path=cli.arguments.add_mnist)
    elif cli.arguments.add_dataset_from_file is not None:
        print("Dataset description file provided: adding these data")
        try:
            with open(cli.arguments.add_dataset_from_file) as json_file:
                data = json.load(json_file)
        except:
            logger.critical("cannot read dataset json file: " + cli.arguments.add_dataset_from_file)
            sys.exit(-1)

        # verify that json file is complete
        for k in ["path", "data_type", "description", "tags", "name"]:
            if k not in data:
                logger.critical("dataset json file corrupted: " + cli.arguments.add_dataset_from_file)

        # dataset path can be defined:
        # - as an absolute path -> take it as it is
        # - as a relative path  -> add the ROOT_DIR in front of it
        # - using an OS environment variable -> transform it
        #
        elements = data["path"].split(os.path.sep)
        if elements[0].startswith("$"):
            # expand OS environment variable
            var = elements[0][1:]
            if var in os.environ:
                var = os.environ[var]
                elements[0] = var
            else:
                logger.info("Unknown env var: " + var)
                elements[0] = ""
        elif elements[0]:
            # p is relative (does not start with /)
            # prepend with topdir
            elements = [environ["ROOT_DIR"]] + elements

        # rebuild the path with these (eventually) new elements
        data["path"] = os.path.join(os.path.sep, *elements)

        # add the dataset to local database (not interactive)
        add_database(interactive=False,
                     path=data["path"],
                     data_type=data["data_type"],
                     description=data["description"],
                     tags=data["tags"],
                     name=data["name"],
                     dataset_parameters=data.get("dataset_parameters")
                     )

    elif cli.arguments.list:
        print('Listing your data available')
        data = dataset_manager.list_my_data(verbose=True)
        if len(data) == 0:
            print('No data has been set up.')
    elif cli.arguments.delete:
        delete_database()
    elif cli.arguments.delete_all:
        delete_all_database()
    elif cli.arguments.delete_mnist:
        delete_database(interactive=False)
    elif cli.arguments.register_training_plan:
        register_training_plan()
    elif cli.arguments.approve_training_plan:
        approve_training_plan()
    elif cli.arguments.reject_training_plan:
        reject_training_plan()
    elif cli.arguments.update_training_plan:
        update_training_plan()
    elif cli.arguments.delete_training_plan:
        delete_training_plan()
    elif cli.arguments.list_training_plans:
        tp_security_manager.list_training_plans(verbose=True)
    elif cli.arguments.view_training_plan:
        view_training_plan()
    elif cli.arguments.start_node:
        # convert to node arguments structure format expected in Round()
        node_args = {
            'gpu': (cli.arguments.gpu_num is not None) or (cli.arguments.gpu is True) or
                   (cli.arguments.gpu_only is True),
            'gpu_num': cli.arguments.gpu_num,
            'gpu_only': (cli.arguments.gpu_only is True)
        }
        launch_node(node_args)


def main():
    """Entry point for the node.
    """
    try:
        launch_cli()
    except KeyboardInterrupt:
        # send error message to researcher via logger.error()
        logger.critical('Operation cancelled by user.')


if __name__ == '__main__':
    main()
