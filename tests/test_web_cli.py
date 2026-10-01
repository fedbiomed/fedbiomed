"""Launchers preserve CLI behavior without coupling API startup to the GUI."""

import argparse
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.web_cli import add_server_arguments
from fedbiomed_gui import cli as gui
from fedbiomed_node_api import cli as api


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    parser = argparse.ArgumentParser()
    add_server_arguments(parser, gui=True)
    args = parser.parse_args([])
    config = SimpleNamespace(root=str(tmp_path))
    # Replace Popen itself; no Gunicorn or Flask process is started.
    popen_mock = MagicMock()
    popen_mock.return_value.__enter__.return_value.wait.return_value = 0
    monkeypatch.setattr(api.subprocess, "Popen", popen_mock)
    assets = tmp_path / "ui" / "build"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text("GUI")
    monkeypatch.setattr(gui, "assets_directory", lambda: assets)
    return args, config, popen_mock, assets


@pytest.mark.parametrize(
    "launcher,target", [(api, "fedbiomed_node_api"), (gui, "fedbiomed_gui.server")]
)
@pytest.mark.parametrize("development", [False, True])
def test_server_command_and_environment(setup, launcher, target, development):
    args, config, popen_mock, _ = setup
    args.development = development
    args.debug = True
    launcher.launch(args, config)
    command = popen_mock.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "flask" if development else "gunicorn"]
    assert target + ".wsgi:app" in command
    assert "shell" not in popen_mock.call_args.kwargs
    env = popen_mock.call_args.kwargs["env"]
    assert env["FBM_NODE_COMPONENT_ROOT"] == config.root
    assert env["DATA_PATH"] == config.root + "/data"
    assert env["FBM_DEBUG"] == "True"
    if development:
        assert "--debug" in command


@pytest.mark.parametrize("development", [False, True])
def test_https_and_paths_with_spaces(setup, tmp_path, development):
    args, config, popen_mock, _ = setup
    args.development = development
    key, cert = tmp_path / "server key.pem", tmp_path / "server cert.pem"
    key.touch()
    cert.touch()
    args.key_file, args.cert_file = str(key), str(cert)
    api.launch(args, config)
    command = popen_mock.call_args.args[0]
    assert str(cert) in command and str(key) in command
    assert ("--cert" if development else "--certfile") in command
    assert ("--key" if development else "--keyfile") in command


@pytest.mark.parametrize("problem", ["data", "partial_tls", "missing_tls"])
def test_invalid_paths_do_not_start_server(setup, problem):
    args, config, popen_mock, _ = setup
    if problem == "data":
        args.data_folder = "/missing/data"
    else:
        args.key_file = "/missing/key"
        args.cert_file = "/missing/cert" if problem == "missing_tls" else None
    with pytest.raises(FedbiomedError):
        api.launch(args, config)
    popen_mock.assert_not_called()


def test_missing_assets_only_block_gui(setup):
    args, config, popen_mock, assets = setup
    (assets / "index.html").unlink()
    with pytest.raises(FedbiomedError, match="fedbiomed-gui --recreate"):
        gui.launch(args, config)
    popen_mock.assert_not_called()
    api.launch(args, config)
    popen_mock.assert_called_once()


@pytest.mark.parametrize("problem", [None, "sources", "yarn", "build"])
def test_recreate_before_start(setup, monkeypatch, problem):
    args, config, popen_mock, assets = setup
    args.recreate = True
    if problem != "sources":
        (assets.parent / "package.json").write_text("{}")
    monkeypatch.setattr(
        gui.shutil, "which", lambda name: None if problem == "yarn" else "/usr/bin/yarn"
    )
    build = MagicMock()
    if problem == "build":
        build.side_effect = [None, subprocess.CalledProcessError(1, ["yarn", "build"])]
    monkeypatch.setattr(gui.subprocess, "run", build)
    if problem:
        with pytest.raises(FedbiomedError):
            gui.launch(args, config)
        popen_mock.assert_not_called()
    else:
        gui.launch(args, config)
        assert [call.args[0] for call in build.call_args_list] == [
            ["/usr/bin/yarn", "install"],
            ["/usr/bin/yarn", "build"],
        ]
        assert all(
            call.kwargs == {"cwd": assets.parent, "check": True}
            for call in build.call_args_list
        )
        popen_mock.assert_called_once()


def test_failed_server_is_reported(setup):
    args, config, popen_mock, _ = setup
    popen_mock.return_value.__enter__.return_value.wait.return_value = 3
    with pytest.raises(FedbiomedError, match="status 3"):
        api.launch(args, config)


def test_missing_server_dependency(setup, monkeypatch):
    """Missing Gunicorn is a server dependency error, not a missing API package."""
    args, config, popen_mock, _ = setup
    monkeypatch.setattr(api.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(FedbiomedError, match="fedbiomed\\[node-api\\]"):
        api.launch(args, config)
    popen_mock.assert_not_called()


@pytest.mark.parametrize(
    "command,package,extra",
    [
        ("gui", "fedbiomed_gui", "gui"),
        ("api", "fedbiomed_node_api", "node-api"),
    ],
)
def test_core_missing_optional_package(command, package, extra, monkeypatch, capsys):
    """Both core commands give installation guidance when their package is absent."""
    from fedbiomed.node import cli

    parser = argparse.ArgumentParser()
    control = (cli.GUIControl if command == "gui" else cli.APIControl)(
        parser.add_subparsers()
    )
    control.initialize()
    load = MagicMock(side_effect=ModuleNotFoundError(name=package))
    monkeypatch.setattr(cli.importlib, "import_module", load)
    args = parser.parse_args([command, "start"])
    with pytest.raises(SystemExit) as error:
        control.forward(args, [])
    assert error.value.code == 2
    load.assert_called_once_with(f"{package}.cli")
    assert f'pip install "fedbiomed[{extra}]"' in capsys.readouterr().err


def test_api_help_does_not_import_gui():
    code = """
import sys
from fedbiomed_node_api.cli import run
try:
    run(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name.startswith('fedbiomed_gui') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_standalone_uses_selected_node(tmp_path, monkeypatch):
    from fedbiomed.node.config import node_component

    root = tmp_path / "node"
    initialize = MagicMock(return_value=SimpleNamespace(root=str(root)))
    monkeypatch.setattr(node_component, "initiate", initialize)
    launch = MagicMock()
    api.run_launcher(launch, argv=["--path", str(root), "--port", "9999"])
    initialize.assert_called_once_with(root=str(root))
    assert launch.call_args.args[0].port == "9999"


def test_core_api_wrapper(monkeypatch):
    """Forward API options and node context; reject the GUI-only rebuild flag."""
    from fedbiomed.node.cli import APIControl

    parser = argparse.ArgumentParser()
    control = APIControl(parser.add_subparsers())
    control.initialize()
    control._context = SimpleNamespace(config=object())
    launch = MagicMock()
    monkeypatch.setattr(api, "launch", launch)
    args = parser.parse_args(["api", "start", "--port", "9000"])
    control.forward(args, [])
    launch.assert_called_once_with(args, control._context.config)
    with pytest.raises(SystemExit):
        parser.parse_args(["api", "start", "--recreate"])
