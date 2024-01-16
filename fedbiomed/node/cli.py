# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Command line user interface for the node component
"""

import argparse
import json
import os
import signal
import sys
import time
import importlib
import functools
import readline

from multiprocessing import Process
from typing import Union, List, Dict
from types import FrameType

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.cli import CommonCLI, CLIArgumentParser



imp_cli_utils = functools.partial(importlib.import_module, "fedbiomed.node.cli_utils")


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


def intro():
    """Print intro for the CLI"""

    print(__intro__)
    print('\t- 🆔 Your node ID:', os.environ['FEDBIOMED_ACTIVE_NODE_ID'], '\n')


def _node_signal_handler(signum: int, frame: Union[FrameType, None]):
    """Signal handler that terminates the process.

    Args:
        signum: Signal number received.
        frame: Frame object received. Currently unused

    Raises:
       SystemExit: Always raised.
    """

    # get the (running) Node object
    global _node

    try:
        if _node and _node.is_connected():
            _node.send_error(ErrorNumbers.FB312,
                             extra_msg = "Node is stopped",
                             broadcast=True)
            time.sleep(2)
            logger.critical("Node stopped in signal_handler, probably node exit on error or user decision (Ctrl C)")
        else:
            # take care of logger level used because message cannot be sent to node
            logger.info("Cannot send error message to researcher (node not initialized yet)")
            logger.info("Node stopped in signal_handler, probably node exit on error or user decision (Ctrl C)")
    finally:
        # give some time to send messages to the researcher
        time.sleep(0.5)
        sys.exit(signum)


def _node_signal_trigger_term() -> None:
    """Triggers a TERM signal to the current process
    """
    os.kill(os.getpid(), signal.SIGTERM)



class DatasetArgumentParser(CLIArgumentParser):
    """Initializes CLI options for dataset actions"""

    def initialize(self):
        """Initializes dataset options for the node CLI"""

        dataset = self._subparser.add_parser(
            "dataset",
            help="Dataset operations"
        )

        # Creates subparser of dataset option
        dataset_subparsers = dataset.add_subparsers()

        # Add option
        add = dataset_subparsers.add_parser(
            "add",
            help="Adds dataset"
        )

        # List option
        list = dataset_subparsers.add_parser(
            "list",
            help="List datasets that are deployed in the node.")

        # Delete option
        delete = dataset_subparsers.add_parser(
            "delete",
            help="Deletes dataset that are deployed in the node.")


        add.add_argument(
            "--mnist",
            "-am",
            metavar="MNIST_DATA_PATH",
            help="Deployes MNIST dataset by downloading form default source to given path.",
            required=False
        )

        add.add_argument(
            "--file",
            "-fl",
            required=False,
            metavar="File that describes the dataset",
            help="File path the dataset file desciptro. This option adds dataset by given file which is has"
                 "cutom format that describes the dataset.")

        delete.add_argument(
            "--all",
            '-a',
            required=False,
            action="store_true",
            help="Removes entire dataset database.")

        delete.add_argument(
            "--only-mnist",
            '-om',
            required=False,
            action="store_true",
            help="Removes only MNIST dataset.")

        add.set_defaults(func=self.add)
        list.set_defaults(func=self.list)
        delete.set_defaults(func=self.delete)


    def add(self, args):
        """Adds datasets"""
        global add_database

        add_database = imp_cli_utils().add_database

        if args.mnist:
            return add_database(interactive=False, path=args.add_mnist)

        if args.file:
            return self._add_dataset_from_file(path=args.file)

        # All operation is handled by CLI utils add_database
        add_database()

    def list(self, unused_args):
        """List datasets

        Args:
          unused_args: Empty arguments since `list` command no positional args.
        """
        dataset_manager = imp_cli_utils().dataset_manager

        print('Listing your data available')
        data = dataset_manager.list_my_data(verbose=True)
        if len(data) == 0:
            print('No data has been set up.')

    def delete(self, args):
        """Deletes datasets"""

        cli_utils = imp_cli_utils()

        if args.all:
            return cli_utils.delete_all_database()

        if args.delete_mnist:
            return cli_utils.delete_database(interactive=False)

        return cli_utils.delete_database()

    def _add_dataset_from_file(self, path):

        print("Dataset description file provided: adding these data")
        try:
            with open(path) as json_file:
                data = json.load(json_file)
        except:
            print(f"Cannot read dataset json file: {path}")
            sys.exit(-1)

        # verify that json file is complete
        for k in ["path", "data_type", "description", "tags", "name"]:
            if k not in data:
                print(f"Dataset json file corrupted: {path}")

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
                     dataset_parameters=data.get("dataset_parameters"))


class TrainingPlanArgumentParser(CLIArgumentParser):

    def initialize(self):

        training_plan = self._subparser.add_parser(
            "training-plan",
            help="CLI operations for TrainingPlans register/list/delete/approve/reject etc."
        )

        training_plan_suparsers = training_plan.add_subparsers()

        update = training_plan_suparsers.add_parser(
            "update", help="Updates training plan"
        )
        update.set_defaults(func=self.update)

        register = training_plan_suparsers.add_parser(
            "register", help="Registers training plans manually by selected file thorugh interactive browser."
        )
        register.set_defaults(func=self.register)

        list = training_plan_suparsers.add_parser(
            "list", help="Lists all saved/registered training plans with their status.")
        list.set_defaults(func=self.list)

        delete = training_plan_suparsers.add_parser(
            "delete", help="Deletes interactively selected training plan from the database.")
        delete.set_defaults(func=self.delete)

        approve = training_plan_suparsers.add_parser(
            "approve", help="Approves interactively selected training plans.")
        approve.set_defaults(func=self.approve)

        reject = training_plan_suparsers.add_parser(
            "reject", help="Rejects interactively selected training plans.")
        reject.set_defaults(func=self.list)

        view = training_plan_suparsers.add_parser(
            "view", help="View interactively selected training plans.")
        view.set_defaults(func=self.view)

    def delete(self):
        """Deletes training plan"""
        delete_training_plan = imp_cli_utils().delete_training_plan
        delete_training_plan()

    def register(self):
        """Registers training plan"""
        register_training_plan = imp_cli_utils().register_training_plan
        register_training_plan()

    def list(self):
        """Lists training plans"""
        tp_security_manager = imp_cli_utils().tp_security_manager
        tp_security_manager.list_training_plans(verbose=True)

    def view(self):
        """Views training plan"""
        view_training_plan = imp_cli_utils().view_training_plan
        view_training_plan()

    def approve(self):
        """Approves training plan"""
        approve_training_plan = imp_cli_utils().approve_training_plan
        approve_training_plan()

    def reject(self):
        """Approves training plan"""
        reject_training_plan = imp_cli_utils().reject_training_plan
        reject_training_plan()

    def update(self):
        """Updates training plan"""
        update_training_plan = imp_cli_utils().update_training_plan
        update_training_plan()


class NodeControl(CLIArgumentParser):

    def initialize(self):
        """Initializes missinon control argument parser"""
        start = self._subparser.add_parser("start", help="Starts the node")
        start.set_defaults(func=self.start)

        start.add_argument(
            "--gpu",
            action="store_true",
            help="Activate GPU usage if the flag is present")

        start.add_argument(
            "--gpu-num",
            "-gn",
            type=int,
            nargs="?",
            required=False,
            default=1,
            help="Number of GPU that is going to be used")

        start.add_argument(
            "--gpu-only",
            "-go",
            action="store_true",
            help="Node performs training only using GPU resources."
                 "This flag automatically activate GPU.")

    def _start_node(self, node_args):
        """Starts the node"""
        cli_utils = imp_cli_utils()

        tp_security_manager = cli_utils.tp_security_manager
        dataset_manager = cli_utils.dataset_manager
        Node = importlib.import_module("fedbiomed.node.node").Node
        environ = importlib.import_module("fedbiomed.node.environ").environ

        try:
            signal.signal(signal.SIGTERM, _node_signal_handler)

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
            _node = Node(dataset_manager=dataset_manager,
                         tp_security_manager=tp_security_manager,
                         node_args=node_args)
            _node.start_messaging(_node_signal_trigger_term)

            logger.info('Starting task manager')
            _node.task_manager()  # handling training tasks in queue

        except FedbiomedError:
            logger.critical("Node stopped.")
            # we may add extra information for the user depending on the error

        except Exception as e:
            # must send info to the researcher (no mqqt should be handled by the previous FedbiomedError)
            _node.send_error(ErrorNumbers.FB300, extra_msg="Error = " + str(e))
            logger.critical("Node stopped.")


    def start(self, args):
        """Starts the node"""

        intro()

        environ = importlib.import_module("fedbiomed.node.environ").environ

        # Define arguments
        node_args = {
            "gpu": (args.gpu is True) or (args.gpu_only is True),
            "gpu_num": args.gpu_num,
            "gpu_only": True if args.gpu_only else False}

        p = Process(target=self._start_node, name='node-' + environ['NODE_ID'], args=(node_args,))
        p.deamon = True
        p.start()

        logger.info("Node started as process with pid = " + str(p.pid))
        try:
            print('To stop press Ctrl + C.')
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            time.sleep(1)
            while p.is_alive():
                logger.info("Terminating process id =" + str(p.pid))
                time.sleep(1)
            logger.info('Exited with code ' + str(p.exitcode))
            sys.exit(0)


class GUIControl(CLIArgumentParser):
    pass


class NodeCLI(CommonCLI):


    _arg_parsers_classes: List[type] = [
        NodeControl,
        DatasetArgumentParser,
        TrainingPlanArgumentParser
    ]
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    def __init__(self):
        super().__init__()

        # Parent parser for parameters that are common for Node CLI actions
        self.initialize()

    @staticmethod
    def config_action(this):
        """Returns CLI argument action for config file name"""
        class ConfigNameAction(argparse.Action):
            """Action for the argument config"""
            def __call__(self, parser, namespace, values, option_string=None):
                print(f'Executing CLI for configraution {values}')
                os.environ["CONFIG_FILE"] = values
                environ = importlib.import_module("fedbiomed.node.environ").environ
                environ.set_environment()
                os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = environ["ID"]
                this.set_environ(environ)

        return ConfigNameAction

    def initialize(self):
        """Initializes node module"""

        self._parser.add_argument(
            "--config",
            "-cf",
            nargs="?",
            action=self.config_action(self),
            default="node_config.ini",
            help="Name of the config file that the CLI will be activated for. Default is 'node_config.ini'.")

        for arg_parser in self._arg_parsers_classes:
            p = arg_parser(self._subparsers)
            p.initialize()
            self._arg_parsers.update({arg_parser.__name__ : p})

# this may be changed on command line or in the config_node.ini
logger.setLevel("DEBUG")
readline.parse_and_bind("tab: complete")
_node = None





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
