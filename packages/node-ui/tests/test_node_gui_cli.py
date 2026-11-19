"""Tests for the Node GUI CLI."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from fedbiomed_gui.cli import main, start_server


@patch("fedbiomed_gui.cli.start_api_server")
def test_start_server_forwards_to_gui_wsgi(start_api_server):
    args = Namespace(recreate=False)
    start_server(args)
    start_api_server.assert_called_once_with(args, "fedbiomed_gui.server.wsgi:app")


@patch("fedbiomed_gui.cli.runpy.run_path")
@patch("fedbiomed_gui.cli.start_api_server")
def test_recreate_calls_hatch_build(start_api_server, run_path):
    build_frontend = MagicMock()
    run_path.return_value = {"build_frontend": build_frontend}
    start_server(Namespace(recreate=True))
    assert run_path.call_args.args[0].name == "hatch_build.py"
    assert build_frontend.call_args.args[0].name == "ui"
    assert build_frontend.call_args.kwargs == {"force": True}
    start_api_server.assert_called_once()


@patch("fedbiomed_gui.cli.start_server")
@patch("sys.argv", ["fedbiomed-ui", "start", "--recreate"])
def test_standalone_command_parses_gui_options(start):
    main()
    assert start.call_args.args[0].recreate is True
