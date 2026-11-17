"""GUI-specific rebuild behavior without importing the optional package."""

import argparse
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from fedbiomed.node.cli import GUIControl


@pytest.fixture
def gui(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    control = GUIControl(parser.add_subparsers())
    control.initialize()
    (tmp_path / "data").mkdir()
    control._context = SimpleNamespace(config=SimpleNamespace(root=str(tmp_path)))
    args = parser.parse_args(["gui", "start", "--recreate"])
    args.path = str(tmp_path)
    package = Mock()
    package.get_frontend_source_dir.return_value = tmp_path / "ui"
    monkeypatch.setattr(
        "fedbiomed.node.cli.importlib",
        SimpleNamespace(import_module=Mock(return_value=package)),
    )
    yarn = Mock(return_value="/usr/bin/yarn")
    monkeypatch.setattr("fedbiomed.node.cli.shutil.which", yarn)
    run = Mock()
    monkeypatch.setattr("fedbiomed.node.cli.subprocess.run", run)
    server = Mock()
    monkeypatch.setattr(control, "_run_server", server)
    return SimpleNamespace(
        control=control,
        parser=parser,
        args=args,
        package=package,
        yarn=yarn,
        run=run,
        server=server,
    )


def test_gui_start_options(gui):
    args = gui.parser.parse_args(
        [
            "gui",
            "start",
            "--recreate",
            "--development",
            "--debug",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--data-folder",
            "/data",
            "--cert-file",
            "/cert.pem",
            "--key-file",
            "/key.pem",
        ]
    )
    assert args.recreate and args.development and args.debug
    assert args.host == "0.0.0.0"
    assert args.port == "9000"


def test_recreate_builds_sources_before_starting(gui):
    order = Mock()
    order.attach_mock(gui.run, "build")
    order.attach_mock(gui.server, "server")
    gui.control.forward(gui.args, [])
    ui = gui.package.get_frontend_source_dir.return_value
    assert gui.run.call_args_list == [
        call(["/usr/bin/yarn", "install"], cwd=ui, check=True),
        call(["/usr/bin/yarn", "build"], cwd=ui, check=True),
    ]
    assert [c[0] for c in order.mock_calls] == ["build", "build", "server"]


@pytest.mark.parametrize(
    "failure,message",
    [
        ("sources", "React sources are unavailable"),
        ("yarn", "Yarn"),
        ("install", "yarn install"),
        ("build", "yarn build"),
    ],
)
def test_recreate_errors_are_readable_and_do_not_start_server(
    gui, failure, message, capsys
):
    if failure == "sources":
        gui.package.get_frontend_source_dir.side_effect = FileNotFoundError(message)
    elif failure == "yarn":
        gui.yarn.return_value = None
    else:
        error = subprocess.CalledProcessError(7, ["yarn", failure])
        gui.run.side_effect = [error] if failure == "install" else [None, error]
    with pytest.raises(SystemExit) as exc:
        gui.control.forward(gui.args, [])
    assert exc.value.code == 1
    assert message.lower() in capsys.readouterr().out.lower()
    gui.server.assert_not_called()


def test_normal_start_needs_neither_sources_nor_yarn(gui):
    gui.args.recreate = False
    gui.control.forward(gui.args, [])
    gui.package.get_frontend_source_dir.assert_not_called()
    gui.yarn.assert_not_called()
    gui.run.assert_not_called()
    gui.server.assert_called_once()
