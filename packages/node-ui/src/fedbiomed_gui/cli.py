"""Command line launcher for the Node GUI server."""

import argparse

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed_node_api.cli import add_start_arguments
from fedbiomed_node_api.cli import start_server as start_api_server

from fedbiomed_gui.server.application import frontend_assets


def start_server(args) -> None:
    """Check the built frontend and launch the composed GUI server."""
    index = frontend_assets() / "index.html"
    if not index.is_file():
        raise FedbiomedError(
            f"The GUI frontend is not built: {index} is missing. "
            f"Run `yarn install && yarn build` in {index.parents[1]}, "
            "then start the GUI again."
        )

    start_api_server(args, "fedbiomed_gui.server.wsgi:app")


def main() -> None:
    """Run the standalone Node GUI command."""
    parser = argparse.ArgumentParser(prog="fedbiomed-ui")
    subparsers = parser.add_subparsers(required=True)
    start = subparsers.add_parser("start", help="Start the Node GUI server")
    add_start_arguments(start)
    start.set_defaults(func=start_server)
    args = parser.parse_args()
    args.func(args)
