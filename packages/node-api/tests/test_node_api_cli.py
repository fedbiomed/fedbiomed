"""Tests for the Node API CLI."""

from argparse import Namespace
from unittest.mock import patch

from fedbiomed_node_api.cli import main, start_server


def arguments(tmp_path, **overrides):
    values = {
        "path": str(tmp_path),
        "data_folder": str(tmp_path),
        "key_file": None,
        "cert_file": None,
        "development": False,
        "debug": False,
        "host": "localhost",
        "port": "8484",
    }
    values.update(overrides)
    return Namespace(**values)


@patch("fedbiomed_node_api.cli.subprocess.Popen")
def test_start_server_uses_gunicorn(popen, tmp_path):
    start_server(arguments(tmp_path))
    command = popen.call_args.args[0]
    assert command[0] == "gunicorn"
    assert command[-1] == "fedbiomed_node_api.wsgi:app"


@patch("fedbiomed_node_api.cli.subprocess.Popen")
def test_start_server_supports_flask_debug(popen, tmp_path):
    start_server(arguments(tmp_path, development=True, debug=True))
    command = popen.call_args.args[0]
    assert command[:3] == ["flask", "--app", "fedbiomed_node_api.wsgi:app"]
    assert "--debug" in command


@patch("fedbiomed_node_api.cli.start_server")
@patch("sys.argv", ["fedbiomed-node-api", "start", "--port", "9000"])
def test_standalone_command_parses_start_options(start):
    main()
    assert start.call_args.args[0].port == "9000"
