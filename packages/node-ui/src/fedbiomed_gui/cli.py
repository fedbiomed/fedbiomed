"""Command line launcher for the Node GUI server."""

import argparse
import runpy
from pathlib import Path

from fedbiomed_node_api.cli import add_start_arguments
from fedbiomed_node_api.cli import start_server as start_api_server


def start_server(args) -> None:
    """Optionally rebuild the frontend and launch the composed GUI server."""
    if args.recreate:
        package_root = Path(__file__).resolve().parents[2]
        build_frontend = runpy.run_path(package_root / "hatch_build.py")[
            "build_frontend"
        ]
        build_frontend(package_root / "ui", force=True)

    start_api_server(args, "fedbiomed_gui.server.wsgi:app")


def main() -> None:
    """Run the standalone Node GUI command."""
    parser = argparse.ArgumentParser(prog="fedbiomed-ui")
    subparsers = parser.add_subparsers(required=True)
    start = subparsers.add_parser("start", help="Start the Node GUI server")
    add_start_arguments(start)
    start.add_argument("--recreate", "-rc", action="store_true")
    start.set_defaults(func=start_server)
    args = parser.parse_args()
    args.func(args)
