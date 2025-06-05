# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import importlib

from fedbiomed.common.constants import DEFAULT_NODE_NAME, DEFAULT_RESEARCHER_NAME
from fedbiomed.common.config import docker_special_case
from fedbiomed.common.cli import CLIArgumentParser, CommonCLI


class UniqueStore(argparse.Action):
    """Argparse action for avoiding having several time the same optional
      argument"""
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not self.default:
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)


class ComponentParser(CLIArgumentParser):
    """Instantiates configuration parser"""

    def initialize(self):
        """Initializes argument parser for creating configuration file."""

        self._parser = self._subparser.add_parser(
            "component",
            help="The helper for generating or updating component configuration files, see `configuration -h`"
            " for more details",
        )

        self._parser.set_defaults(func=self.default)

        # Common parser to register common arguments for create and refresh
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument(
            "-p",
            "--path",
            action=UniqueStore,
            metavar="COMPONENT_PATH",
            type=str,
            nargs="?",
            required=False,
            help="Path to specificy where Fed-BioMed component will be intialized.",
        )

        common_parser.add_argument(
            "-c",
            "--component",
            metavar="COMPONENT_TYPE[ NODE|RESEARCHER ]",
            type=str,
            nargs="?",
            required=True,
            help="Component type NODE or RESEARCHER",
        )

        # Create sub parser under `configuration` command
        component_sub_parsers = self._parser.add_subparsers()

        create = component_sub_parsers.add_parser(
            "create",
            parents=[common_parser],
            help="Creates component folder for the specified component if it does not exist. "
            "If the component folder exists, leave it unchanged",
        )

        create.add_argument(
            "-eo",
            "--exist-ok",
            action="store_true",
            help="Creates configuration only if there isn't an existing one",
        )

        create.set_defaults(func=self.create)

    def _get_component_instance(self, path: str, component: str):
        """Gets component"""
        if component.lower() == "node":
            config_node = importlib.import_module("fedbiomed.node.config")
            _component = config_node.node_component
        elif component.lower() == "researcher":
            os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = path
            config_researcher = importlib.import_module(
                "fedbiomed.researcher.config"
            )
            _component = config_researcher.researcher_component
        else:
            print(f"Undefined component type {component}")
            sys.exit(101)

        return _component

    def create(self, args):
        """CLI Handler for creating configuration file and assets for given component
        """
        if args.component is None:
            CommonCLI.error("Error: bad command line syntax")

        if not args.path:
            if args.component.lower() == "researcher":
                component_path = os.path.join(os.getcwd(), DEFAULT_RESEARCHER_NAME)
            else:
                component_path = os.path.join(os.getcwd(), DEFAULT_NODE_NAME)
        else:
            component_path = args.path

        # Researcher specific case ----------------------------------------------------
        # This is a special case since researcher import
        if args.component.lower() == "researcher":
            if DEFAULT_RESEARCHER_NAME in component_path and \
                os.path.isdir(component_path) and \
                not docker_special_case(component_path):
                if not args.exist_ok:
                    CommonCLI.error(
                        f"Default component is already existing. In the directory {component_path} "
                        "please remove existing one to re-initiate"
                    )
                else:
                    CommonCLI.success(
                        "Component is already existing. Using existing component."
                    )
                    return
            else:
                self._get_component_instance(component_path, args.component)
                return
        else:
            component = self._get_component_instance(component_path, args.component)
            # Overwrite force configuration file
            if component.is_component_existing(component_path):
                if not args.exist_ok:
                    CommonCLI.error(
                        f"Component is already existing in the directory `{component_path}`. To ignore "
                       "this error please execute component creation using `--exist-ok`"
                )
                else:
                    CommonCLI.success(
                        "Component is already existing. Using existing component."
                    )
                    return

            component.initiate(component_path)

        CommonCLI.success(f"Component has been initialized in {component_path}")



cli = CommonCLI()
cli.initialize_optional()

# Initialize configuration parser
configuration_parser = ComponentParser(cli.subparsers)
configuration_parser.initialize()

# Add node and researcher options
node_p = cli.subparsers.add_parser(
    "node", add_help=False, help="Command for managing Node component"
)
researcher_p = cli.subparsers.add_parser(
    "researcher", add_help=False, help="Command for managing Researcher component"
)


def node(args):
    """Forwards node CLI"""
    NodeCLI = importlib.import_module("fedbiomed.node.cli").NodeCLI
    cli = NodeCLI()
    cli.parse_args(args)


def researcher(args):
    """Forwards researcher CLI"""
    ResearcherCLI = importlib.import_module("fedbiomed.researcher.cli").ResearcherCLI
    cli = ResearcherCLI()
    cli.parse_args(args)


node_p.set_defaults(func=node)
researcher_p.set_defaults(func=researcher)


def run():
    """Runs the CLI"""
    # This part executes know arguments
    args, extra = cli.parser.parse_known_args()
    # Forward arguments to Node or Researcher CLI
    if hasattr(args, "func") and args.func in [node, researcher]:
        args.func(extra)
    elif hasattr(args, "func"):
        args.func(args)
    # If there is no command provided
    else:
        cli.parse_args(["--help"])

if __name__ == "__main__":
        run()
