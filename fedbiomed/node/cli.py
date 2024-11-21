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
import subprocess
import importlib.util

from multiprocessing import Process
from typing import Union, List, Dict
from types import FrameType
from pathlib import Path


from fedbiomed.common.constants import ErrorNumbers, ComponentType
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.cli import (
    CommonCLI,
    CLIArgumentParser,
    ConfigNameAction,
)

# Partial function to import CLI utils that frequently used in this module
imp_cli_utils = functools.partial(importlib.import_module, "fedbiomed.node.cli_utils")


# Please use following code genereate similar intro
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
    """Prints intro for the CLI"""

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



def start_node(node_args):
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
        logger.info('Starting node to node router')
        _node.start_protocol()
        logger.info('Starting task manager')
        _node.task_manager()  # handling training tasks in queue

    except FedbiomedError as exp:
        logger.critical(f"Node stopped. {exp}")
        # we may add extra information for the user depending on the error

    except Exception as exp:
        # must send info to the researcher (no mqqt should be handled
        # by the previous FedbiomedError)
        _node.send_error(ErrorNumbers.FB300, extra_msg="Error = " + str(exp))
        logger.critical(f"Node stopped. {exp}")



class DatasetArgumentParser(CLIArgumentParser):
    """Initializes CLI options for dataset actions"""

    def initialize(self):
        """Initializes dataset options for the node CLI"""

        self._parser = self._subparser.add_parser(
            "dataset",
            help="Dataset operations"
        )
        self._parser.set_defaults(func=self.default)

        # Creates subparser of dataset option
        dataset_subparsers = self._parser.add_subparsers()


        # Add option
        add = dataset_subparsers.add_parser(
            "add",
            help="Adds dataset"
        )

        # List option
        list_ = dataset_subparsers.add_parser(
            "list",
            help="List datasets that are deployed in the node.")

        # Delete option
        delete = dataset_subparsers.add_parser(
            "delete",
            help="Deletes dataset that are deployed in the node.")


        add.add_argument(
            "--mnist",
            "-m",
            metavar="MNIST_DATA_PATH",
            help="Deploys MNIST dataset by downloading form default source to given path.",
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
            "--mnist",
            '-m',
            required=False,
            action="store_true",
            help="Removes MNIST dataset.")

        add.set_defaults(func=self.add)
        list_.set_defaults(func=self.list)
        delete.set_defaults(func=self.delete)

    def add(self, args):
        """Adds datasets"""

        global add_database

        add_database = imp_cli_utils().add_database

        if args.mnist:
            return add_database(interactive=False, path=args.mnist)

        if args.file:
            return self._add_dataset_from_file(path=args.file)

        # All operation is handled by CLI utils add_database
        return add_database()

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

        if args.mnist:
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
            environ = importlib.import_module("fedbiomed.node.environ").environ
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
    """Argument parser for training-plan operations"""

    def initialize(self):

        self._parser = self._subparser.add_parser(
            "training-plan",
            help="CLI operations for TrainingPlans register/list/delete/approve/reject etc."
        )

        training_plan_suparsers = self._parser.add_subparsers()
        self._parser.set_defaults(func=self.default)


        common_reject_approve = argparse.ArgumentParser(add_help=False)
        common_reject_approve.add_argument(
            '--id',
            type=str,
            nargs='?',
            required=False,
            help='ID of the training plan that will be processed.'
        )


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
            "delete",
            parents=[common_reject_approve],
            help="Deletes interactively selected training plan from the database.")
        delete.set_defaults(func=self.delete)

        approve = training_plan_suparsers.add_parser(
            "approve",
            parents=[common_reject_approve],
            help="Approves interactively selected training plans.")
        approve.set_defaults(func=self.approve)

        reject = training_plan_suparsers.add_parser(
            "reject",
            parents=[common_reject_approve],
            help="Rejects interactively selected training plans.")

        reject.add_argument(
            "--notes",
            type=str,
            nargs="?",
            required=False,
            default="No notes provided.",
            help="Note to explain why training plan is rejected."
        )
        reject.set_defaults(func=self.reject)

        view = training_plan_suparsers.add_parser(
            "view", help="View interactively selected training plans.")
        view.set_defaults(func=self.view)

    def delete(self, args):
        """Deletes training plan"""
        delete_training_plan = imp_cli_utils().delete_training_plan
        delete_training_plan(id=args.id)

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

    def approve(self, args):
        """Approves training plan"""
        approve_training_plan = imp_cli_utils().approve_training_plan
        approve_training_plan(id=args.id)

    def reject(self, args):
        """Approves training plan"""
        reject_training_plan = imp_cli_utils().reject_training_plan
        reject_training_plan(id=args.id, notes=args.notes)

    def update(self):
        """Updates training plan"""
        update_training_plan = imp_cli_utils().update_training_plan
        update_training_plan()


class NodeControl(CLIArgumentParser):
    """CLI argument parser for starting the node"""

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


    def start(self, args):
        """Starts the node"""

        global start_node

        intro()
        environ = importlib.import_module("fedbiomed.node.environ").environ


        # Define arguments
        node_args = {
            "gpu": (args.gpu is True) or (args.gpu_only is True),
            "gpu_num": args.gpu_num,
            "gpu_only": True if args.gpu_only else False}

        p = Process(target=start_node, name='node-' + environ['NODE_ID'], args=(node_args,))
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


    def initialize(self):
        """Initializes GUI commands"""
        self._parser = self._subparser.add_parser(
            "gui", add_help=False, help="Action to manage Node user interface"
        )
        self._parser.set_defaults(func=self.forward)

        gui_subparsers = self._parser.add_subparsers()
        start = gui_subparsers.add_parser('start')


        start.add_argument(
            "--data-folder",
           "-df",
            type=str,
            nargs="?",
            default="data",  # data folder in root directory
            required=False)

        start.add_argument(
            "--cert-file",
            "-cf",
            type=str,
            nargs="?",
            required=False,
            help="Name of the certificate to use in order to enable HTTPS. "
                 "If cert file doesn't exist script will raise an error.")

        start.add_argument(
            "--key-file",
            "-kf",
            type=str,
            nargs="?",
            required=False,
            help="Name of the private key for the SSL certificate. "
                 "If the key file doesn't exist, the script will raise an error.")

        start.add_argument(
            "--port",
            "-p",
            type=str,
            nargs="?",
            default="8484",
            required=False,
            help="HTTP port that GUI will be served. Default is `8484`")

        start.add_argument(
            "--host",
            "-ho",
            type=str,
            default="localhost",
            nargs="?",
            required=False,
            help="HTTP port that GUI will be served. Default is `8484`")

        start.add_argument(
            "--debug",
            "-dbg",
            action="store_true",
            required=False,
            help="HTTP port that GUI will be served. Default is `8484`")

        start.add_argument(
            "--recreate",
            "-rc",
            action="store_true",
            required=False,
            help="Re-creates gui build")

        start.add_argument(
            "--development",
            "-dev",
            action="store_true",
            required=False,
            help="If it is set, GUI will start in development mode."
        )

        start.set_defaults(func=self.forward)



    def forward(self, args, extra_args):
        """Forwards gui commands to ./script/fedbiomed_gui Extra arguments

        TODO: Implement argument GUI parseing and execution
        """

        fedbiomed_root = os.path.abspath(args.config)

        os.environ.update({
            "DATA_PATH": os.path.abspath(args.data_folder),
            "FBM_NODE_COMPONENT_ROOT": fedbiomed_root,
        })
        current_env = os.environ.copy()

        if args.key_file and args.cert_file:
            certificate = ["--keyfile", args.key_file, "--certfile", args.cert_file ]
        else:
            certificate = []

        host_port = ["--host", args.host, "--port", args.port]
        if args.development:
            command = [
                "FLASK_ENV=development",
                f"FLASK_APP={gui_server.__file__}",
                "flask",
                "run",
                *host_port,
                *certificate
            ]
        else:
            command = [
                "gunicorn",
                "--workers",
                "1",
                # str(os.cpu_count()),
                *certificate,
                "-b",
                f"{args.host}:{args.port}",
                "--access-logfile",
                "-",
                "fedbiomed.gui.server.wsgi:app"
            ]

        try:
            with subprocess.Popen(" ".join(command), env=current_env, shell=True) as proc:
                proc.wait()
        except Exception as e:
            print(e)

class NodeCLI(CommonCLI):

    _arg_parsers_classes: List[type] = [
        NodeControl,
        DatasetArgumentParser,
        TrainingPlanArgumentParser,
        GUIControl
    ]
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    def __init__(self):
        super().__init__()

        self._parser.prog = "fedbiomed_run node"
        self.description = f"{__intro__} \nA CLI app for fedbiomed node component."
        # Parent parser for parameters that are common for Node CLI actions
        self.initialize()

    def initialize(self):
        """Initializes node module"""


        class ConfigNameActionNode(ConfigNameAction):

            _this = self
            _component = ComponentType.NODE

            def import_environ(self, config_file: str | None = None) -> 'fedbiomed.node.environ.Environ':
                """Imports dynamically node environ object"""

                if config_file:
                    os.environ["FBM_NODE_COMPONENT_ROOT"] = os.path.join(config_file)
                else:
                    print("Component is not specified: Using 'fbm-researcher' in current working directory...")
                    os.environ["FBM_NODE_COMPONENT_ROOT"] = \
                        os.path.join(os.getcwd(), 'fbm-node')

                return importlib.import_module("fedbiomed.node.environ").environ

        self._parser.add_argument(
            "--config",
            "-c",
            nargs="?",
            action=ConfigNameActionNode,
            default="config_node.ini",
            help="Name of the config file that the CLI will be activated for. Default is 'config_node.ini'.")

        super().initialize()


if __name__ == '__main__':
    cli = NodeCLI()
    cli.parse_args()
