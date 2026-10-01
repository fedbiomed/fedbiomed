"""Launch the Node HTTP API independently of frontend assets."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.web_cli import add_server_arguments


def serve(args, config, application):
    data = Path(args.data_folder or Path(config.root) / NODE_DATA_FOLDER).resolve()
    if not data.is_dir():
        raise FedbiomedError(f"path {data} is not a folder. Aborting")
    if bool(args.key_file) != bool(args.cert_file):
        raise FedbiomedError("Provide both --cert-file and --key-file for HTTPS.")
    for filename in (args.key_file, args.cert_file):
        if filename and not Path(filename).is_file():
            raise FedbiomedError(f"HTTPS file does not exist: {filename}")
    env = os.environ.copy()
    env.update(
        DATA_PATH=str(data), FBM_NODE_COMPONENT_ROOT=str(Path(config.root).resolve())
    )
    if args.debug:
        env["FBM_DEBUG"] = "True"
    module = "flask" if args.development else "gunicorn"
    if importlib.util.find_spec(module) is None:
        raise FedbiomedError(
            f'Missing server dependency {module}. Install: python -m pip install "fedbiomed[node-api]"'
        )
    command = [sys.executable, "-m", module]
    if args.development:
        command += ["--app", application]
        if args.debug:
            command += ["--debug"]
        command += ["run", "--host", args.host, "--port", str(args.port)]
        if args.cert_file:
            command += ["--cert", args.cert_file, "--key", args.key_file]
    else:
        command += [
            "--workers",
            "1",
            "-b",
            f"{args.host}:{args.port}",
            "--access-logfile",
            "-",
        ]
        if args.cert_file:
            command += ["--certfile", args.cert_file, "--keyfile", args.key_file]
        command += [application]
    try:
        with subprocess.Popen(command, env=env) as process:
            try:
                status = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                return
    except OSError as exc:
        raise FedbiomedError(f"Could not start the server: {exc}") from exc
    if status:
        raise FedbiomedError(f"Server exited with status {status}.")


def launch(args, config):
    """Launch using arguments and node configuration already parsed by run_launcher."""
    serve(args, config, "fedbiomed_node_api.wsgi:app")


def run_launcher(launcher, *, gui=False, argv=None):
    """Shared standalone parsing/setup for the API and GUI launch callbacks."""
    parser = argparse.ArgumentParser(
        prog="fedbiomed-gui" if gui else "fedbiomed-node-api"
    )
    parser.add_argument("--path", default="fbm-node", help="Node component directory")
    add_server_arguments(parser, gui=gui)
    args = parser.parse_args(argv)
    from fedbiomed.node.config import node_component

    try:
        config = node_component.initiate(root=str(Path(args.path).resolve()))
        launcher(args, config)
    except FedbiomedError as exc:
        parser.error(str(exc))


def run(argv=None):
    """Console entry point selecting the API callback without importing GUI."""
    run_launcher(launch, argv=argv)


if __name__ == "__main__":
    run()
