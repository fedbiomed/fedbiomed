"""Command line launcher for the Node API server."""

import argparse
import os
import subprocess

from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.common.exceptions import FedbiomedError


def add_start_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by API and GUI server commands."""
    parser.add_argument("--path", default="fbm-node")
    parser.add_argument("--data-folder", "-df", default="")
    parser.add_argument("--cert-file", "-cf")
    parser.add_argument("--key-file", "-kf")
    parser.add_argument("--port", "-p", default="8484")
    parser.add_argument("--host", "-ho", default="localhost")
    parser.add_argument("--debug", "-dbg", action="store_true")
    parser.add_argument("--development", "-dev", action="store_true")


def start_server(args, application="fedbiomed_node_api.wsgi:app") -> None:
    """Validate server options and run Flask or Gunicorn."""
    node_root = os.path.abspath(args.path)
    data_folder = (
        os.path.abspath(args.data_folder)
        if args.data_folder
        else os.path.join(node_root, NODE_DATA_FOLDER)
    )
    if not os.path.isdir(data_folder):
        raise FedbiomedError(f"path {data_folder} is not a folder. Aborting")

    env = os.environ.copy()
    env.update({"DATA_PATH": data_folder, "FBM_NODE_COMPONENT_ROOT": node_root})
    certificate = []
    if args.key_file and args.cert_file:
        certificate = ["--keyfile", args.key_file, "--certfile", args.cert_file]

    if args.development:
        command = [
            "flask",
            "--app",
            application,
            "run",
            "--host",
            args.host,
            "--port",
            args.port,
            *certificate,
        ]
        env["FLASK_ENV"] = "development"
        if args.debug:
            command.append("--debug")
    else:
        command = [
            "gunicorn",
            "--workers",
            "1",
            *certificate,
            "-b",
            f"{args.host}:{args.port}",
            "--access-logfile",
            "-",
            application,
        ]

    try:
        with subprocess.Popen(command, env=env) as process:
            try:
                process.wait()
            except KeyboardInterrupt:
                process.wait()
    except OSError as exc:
        raise FedbiomedError(f"Could not start the Node server: {exc}") from exc


def main() -> None:
    """Run the standalone Node API command."""
    parser = argparse.ArgumentParser(prog="fedbiomed-node-api")
    subparsers = parser.add_subparsers(required=True)
    start = subparsers.add_parser("start", help="Start the Node API server")
    add_start_arguments(start)
    start.set_defaults(func=start_server)
    args = parser.parse_args()
    args.func(args)
