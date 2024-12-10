# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import traceback
import importlib
from fedbiomed.common.cli import CommonCLI
import os


cli = CommonCLI()
cli.initialize_optional()


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
