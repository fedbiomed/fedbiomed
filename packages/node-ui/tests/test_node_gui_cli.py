"""Tests for the Node GUI CLI."""

from argparse import Namespace
from unittest.mock import patch

import pytest
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed_gui.cli import main, start_server


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr("fedbiomed_gui.cli.frontend_assets", lambda: tmp_path / "build")
    return tmp_path / "build"


@patch("fedbiomed_gui.cli.start_api_server")
def test_start_server_launches_gui_when_frontend_is_built(start_api_server, assets):
    assets.mkdir()
    (assets / "index.html").write_text("")
    args = Namespace()
    start_server(args)
    start_api_server.assert_called_once_with(args, "fedbiomed_gui.server.wsgi:app")


@patch("fedbiomed_gui.cli.start_api_server")
def test_missing_frontend_explains_how_to_build(start_api_server, assets):
    with pytest.raises(FedbiomedError, match="yarn install && yarn build"):
        start_server(Namespace())
    start_api_server.assert_not_called()


@patch("fedbiomed_gui.cli.start_server")
@patch("sys.argv", ["fedbiomed-ui", "start", "--port", "5001"])
def test_standalone_command_accepts_server_options(start):
    main()
    assert start.call_args.args[0].port == "5001"
