# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Command line user interface for the node component
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from fedbiomed.common.cli import (
    CLIArgumentParser,
    CommonCLI,
    ComponentDirectoryAction,
)
from fedbiomed.common.constants import NODE_DATA_FOLDER, ComponentType
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.node.cli_utils import (
    add_database,
    approve_training_plan,
    delete_all_database,
    delete_database,
    delete_training_plan,
    register_training_plan,
    reject_training_plan,
    update_training_plan,
    view_training_plan,
)
from fedbiomed.node.config import node_component
from fedbiomed.node.node import Node
from fedbiomed.node.node_pm import NodeProcessManager

# Please use following code generate similar intro
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
    print("\t- 🆔 Your node ID:", os.environ["FEDBIOMED_ACTIVE_NODE_ID"], "\n")


class DatasetArgumentParser(CLIArgumentParser):
    """Initializes CLI options for dataset actions"""

    _node: Node
    _mnist_path = None

    def initialize(self):
        """Initializes dataset options for the node CLI"""

        self._parser = self._subparser.add_parser("dataset", help="Dataset operations")
        self._parser.set_defaults(func=self.default)

        # Creates subparser of dataset option
        dataset_subparsers = self._parser.add_subparsers()

        # Add option
        add = dataset_subparsers.add_parser("add", help="Adds dataset")
        add.set_defaults(func=self.add)

        # List option
        list_ = dataset_subparsers.add_parser(
            "list", help="List datasets that are deployed in the node."
        )

        # Delete option
        delete = dataset_subparsers.add_parser(
            "delete", help="Deletes dataset that are deployed in the node."
        )

        add.add_argument(
            "--mnist",
            "-m",
            metavar="MNIST_DATA_PATH",
            help="Deploys MNIST dataset by downloading form default source to given path.",
            nargs="?",
            type=str,
            required=False,
            default="",
        )
        self._mnist_path = add

        add.add_argument(
            "--file",
            "-fl",
            required=False,
            metavar="File that describes the dataset",
            help="File path the dataset file description. This option adds dataset by given file which has"
            "custom format that describes the dataset.",
        )

        delete.add_argument(
            "--all",
            "-a",
            required=False,
            action="store_true",
            help="Removes entire dataset database.",
        )

        delete.add_argument(
            "--mnist",
            "-m",
            required=False,
            action="store_true",
            help="Removes MNIST dataset.",
        )

        list_.set_defaults(func=self.list)
        delete.set_defaults(func=self.delete)

    def add(self, args):
        """Adds datasets"""

        if args.mnist != "":
            if args.mnist is None:
                mnist_path = os.path.join(self._node.config.root, NODE_DATA_FOLDER)
            else:
                mnist_path = args.mnist
            return add_database(
                self._node.dataset_manager, interactive=False, path=mnist_path
            )

        if args.file:
            return self._add_dataset_from_file(path=args.file)

        # All operation is handled by CLI utils add_database
        return add_database(self._node.dataset_manager)

    def list(self, unused_args):
        """List datasets

        Args:
          unused_args: Empty arguments since `list` command no positional args.
        """
        print("Listing your data available")
        data = self._node.dataset_manager.list_my_datasets(verbose=True)
        if len(data) == 0:
            print("No data has been set up.")

    def delete(self, args):
        """Deletes datasets"""

        if args.all:
            return delete_all_database(self._node.dataset_manager)

        if args.mnist:
            return delete_database(self._node.dataset_manager, interactive=False)

        return delete_database(self._node.dataset_manager)

    def _add_dataset_from_file(self, path):
        print("Dataset description file provided: adding these data")
        try:
            with open(path) as json_file:
                data = json.load(json_file)
        except:  # noqa: E722
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
            elements = [self._node.config.root] + elements

        # rebuild the path with these (eventually) new elements
        data["path"] = os.path.join(os.path.sep, *elements)

        # add the dataset to local database (not interactive)
        add_database(
            self._node.dataset_manager,
            interactive=False,
            path=data["path"],
            data_type=data["data_type"],
            description=data["description"],
            tags=data["tags"],
            name=data["name"],
            dataset_parameters=data.get("dataset_parameters"),
        )


class TrainingPlanArgumentParser(CLIArgumentParser):
    """Argument parser for training-plan operations"""

    _node: Node

    def initialize(self):
        self._parser = self._subparser.add_parser(
            "training-plan",
            help="CLI operations for TrainingPlans register/list/delete/approve/reject etc.",
        )

        training_plan_suparsers = self._parser.add_subparsers()
        self._parser.set_defaults(func=self.default)

        common_reject_approve = argparse.ArgumentParser(add_help=False)
        common_reject_approve.add_argument(
            "--id",
            type=str,
            nargs="?",
            required=False,
            help="ID of the training plan that will be processed.",
        )

        update = training_plan_suparsers.add_parser(
            "update", help="Updates training plan"
        )
        update.set_defaults(func=self.update)

        register = training_plan_suparsers.add_parser(
            "register",
            help="Registers training plans manually by selected file through interactive browser.",
        )
        register.set_defaults(func=self.register)

        list = training_plan_suparsers.add_parser(
            "list", help="Lists all saved/registered training plans with their status."
        )
        list.set_defaults(func=self.list)

        delete = training_plan_suparsers.add_parser(
            "delete",
            parents=[common_reject_approve],
            help="Deletes interactively selected training plan from the database.",
        )
        delete.set_defaults(func=self.delete)

        approve = training_plan_suparsers.add_parser(
            "approve",
            parents=[common_reject_approve],
            help="Approves interactively selected training plans.",
        )
        approve.set_defaults(func=self.approve)

        reject = training_plan_suparsers.add_parser(
            "reject",
            parents=[common_reject_approve],
            help="Rejects interactively selected training plans.",
        )

        reject.add_argument(
            "--notes",
            type=str,
            nargs="?",
            required=False,
            default="No notes provided.",
            help="Note to explain why training plan is rejected.",
        )
        reject.set_defaults(func=self.reject)

        view = training_plan_suparsers.add_parser(
            "view", help="View interactively selected training plans."
        )
        view.set_defaults(func=self.view)

    def delete(self, args):
        """Deletes training plan"""
        delete_training_plan(self._node.tp_security_manager, id=args.id)

    def register(self):
        """Registers training plan"""
        register_training_plan(self._node.tp_security_manager)

    def list(self):
        """Lists training plans"""
        self._node.tp_security_manager.list_training_plans(verbose=True)

    def view(self):
        """Views training plan"""
        view_training_plan(self._node.tp_security_manager)

    def approve(self, args):
        """Approves training plan"""
        approve_training_plan(self._node.tp_security_manager, id=args.id)

    def reject(self, args):
        """Approves training plan"""
        reject_training_plan(
            self._node.tp_security_manager, id=args.id, notes=args.notes
        )

    def update(self):
        """Updates training plan"""
        update_training_plan(self._node.tp_security_manager)


class NodeControl(CLIArgumentParser):
    """CLI argument parser for starting the node"""

    _node: Node

    def initialize(self):
        """Initializes missing control argument parser"""
        start = self._subparser.add_parser("start", help="Starts the node")
        start.set_defaults(func=self.start)

        start.add_argument(
            "--gpu",
            action="store_true",
            help="Activate GPU usage if the flag is present",
        )

        start.add_argument(
            "--gpu-num",
            "-gn",
            type=int,
            nargs="?",
            required=False,
            default=1,
            help="Number of GPU that is going to be used",
        )

        start.add_argument(
            "--gpu-only",
            "-go",
            action="store_true",
            help="Node performs training only using GPU resources."
            "This flag automatically activate GPU.",
        )

        start.add_argument(
            "--debug",
            "-D",
            action="store_true",
            required=False,
            help="Activate debug mode for the Node. Default is `False`",
        )

        start.add_argument(
            "--background",
            "-b",
            action="store_true",
            required=False,
            help="Start the node in the background. Default is `False`",
        )

    def start(self, args):
        """Starts the node"""
        intro()

        node_args = {
            "gpu": (args.gpu is True) or (args.gpu_only is True),
            "gpu_num": args.gpu_num,
            "gpu_only": True if args.gpu_only else False,
            "debug": True if args.debug else False,
            "background": True if args.background else False,
        }

        node_process_manager = NodeProcessManager(self._node.config)

        try:
            print("To stop press Ctrl + C.")
            node_process_manager.start(
                node_args=node_args,
                background=node_args.get("background", False),
                actor={"source": "cli"},
            )
        except KeyboardInterrupt:
            node_process_manager.stop(
                actor={"source": "cli"},
                reason="keyboard_interrupt",
            )
            sys.exit(0)


class GUIControl(CLIArgumentParser):
    _node: Node

    def initialize(self):
        """Initializes GUI commands"""
        self._parser = self._subparser.add_parser(
            "gui",  # add_help=False,
            help="Action to manage Node user interface",
        )

        gui_subparsers = self._parser.add_subparsers(title="start GUI")
        start = gui_subparsers.add_parser(
            "start", help="Launch the server (defaults on localhost:8484)"
        )

        start.set_defaults(func=self.forward)

        start.add_argument(
            "--data-folder",
            "-df",
            type=str,
            nargs="?",
            default="",  # data folder in root directory
            required=False,
        )

        start.add_argument(
            "--cert-file",
            "-cf",
            type=str,
            nargs="?",
            required=False,
            help="Name of the certificate to use in order to enable HTTPS. "
            "If cert file doesn't exist script will raise an error.",
        )

        start.add_argument(
            "--key-file",
            "-kf",
            type=str,
            nargs="?",
            required=False,
            help="Name of the private key for the SSL certificate. "
            "If the key file doesn't exist, the script will raise an error.",
        )

        start.add_argument(
            "--port",
            "-p",
            type=str,
            nargs="?",
            default="8484",
            required=False,
            help="HTTP port that GUI will be served. Default is `8484`",
        )

        start.add_argument(
            "--host",
            "-ho",
            type=str,
            default="localhost",
            nargs="?",
            required=False,
            help="HTTP port that GUI will be served. Default is `127.0.0.1` (localhost)",
        )

        start.add_argument(
            "--debug",
            "-dbg",
            action="store_true",
            required=False,
            help="HTTP port that GUI will be served. Default is `8484`",
        )

        start.add_argument(
            "--recreate",
            "-rc",
            action="store_true",
            required=False,
            help="Re-creates gui build",
        )

        start.add_argument(
            "--development",
            "-dev",
            action="store_true",
            required=False,
            help="If it is set, GUI will start in development mode.",
        )

    def forward(self, args: argparse.Namespace, extra_args):
        """Launches Fed-BioMed Node GUI

        Args:
            args: parser argument's namespace
        """

        fedbiomed_root = os.path.abspath(args.path)

        if args.data_folder == "":
            data_folder = os.path.join(self._node.config.root, NODE_DATA_FOLDER)
        else:
            data_folder = os.path.abspath(args.data_folder)
        if not os.path.isdir(data_folder):
            raise FedbiomedError(f"path {data_folder} is not a folder. Aborting")
        os.environ.update(
            {
                "DATA_PATH": data_folder,
                "FBM_NODE_COMPONENT_ROOT": fedbiomed_root,
            }
        )
        current_env = os.environ.copy()

        if args.key_file and args.cert_file:
            certificate = ["--keyfile", args.key_file, "--certfile", args.cert_file]
        else:
            certificate = []

        fedbiomed_gui = importlib.import_module("fedbiomed_gui")
        server_app = Path(fedbiomed_gui.__file__).parent  # type: ignore[arg-type]
        print("path to server", server_app)

        host_port = ["--host", args.host, "--port", args.port]
        if args.development:
            command = [
                "FLASK_ENV=development",
                f"FLASK_APP={os.path.join(server_app, 'server', 'wsgi.py')}",
                "flask",
                "run",
                *host_port,
                *certificate,
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
                "fedbiomed_gui.server.wsgi:app",
            ]

        try:
            with subprocess.Popen(
                " ".join(command), env=current_env, shell=True
            ) as proc:
                proc.wait()
        except Exception as e:
            print(e)


class NodeCLI(CommonCLI):
    _arg_parsers_classes: List[type] = [
        NodeControl,
        DatasetArgumentParser,
        TrainingPlanArgumentParser,
        GUIControl,
    ]
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    def __init__(self):
        super().__init__()

        self._parser.prog = "fedbiomed node"
        self.description = f"{__intro__} \nA CLI app for fedbiomed node component."
        # Parent parser for parameters that are common for Node CLI actions
        self.initialize()

    def initialize(self):
        """Initializes node module"""

        class ComponentDirectoryActionNode(ComponentDirectoryAction):
            _this = self
            _component = ComponentType.NODE

            def set_component(self, component_dir: str | None = None) -> None:
                """Create node instance"""
                if component_dir:
                    component_dir = os.path.abspath(component_dir)
                    os.environ["FBM_NODE_COMPONENT_ROOT"] = component_dir
                else:
                    print(
                        "Component is not specified: Using 'fbm-node' in current working directory..."
                    )
                    component_dir = os.path.join(os.getcwd(), "fbm-node")
                    os.environ["FBM_NODE_COMPONENT_ROOT"] = component_dir
                config = node_component.initiate(root=component_dir)
                self._this.config = config
                node = Node(config)
                # Set node object to make it accessible
                ComponentDirectoryActionNode._this._node = node
                os.environ[f"FEDBIOMED_ACTIVE_{self._component.name}_ID"] = config.get(
                    "default", "id"
                )

                # Set node in all subparsers
                for (
                    _,
                    parser,
                ) in ComponentDirectoryActionNode._this._arg_parsers.items():
                    parser._node = node

        super().initialize()

        self._parser.add_argument(
            "--path",
            "-p",
            nargs="?",
            action=ComponentDirectoryActionNode,
            default="fbm-node",
            help="The path were component is located. It can be absolute or "
            "relative to the path where CLI is executed.",
        )


if __name__ == "__main__":
    cli = NodeCLI()
    cli.parse_args()
