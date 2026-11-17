# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""API and GUI launchers use optional imports and the active interpreter."""

import argparse
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fedbiomed.node.cli import APIControl, GUIControl, NodeCLI


@pytest.fixture(params=[APIControl, GUIControl], ids=["api", "gui"])
def launcher(request, tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    control = request.param(parser.add_subparsers())
    control.initialize()
    root = tmp_path / "node with spaces"
    data = root / "data"
    data.mkdir(parents=True)
    control._context = SimpleNamespace(config=SimpleNamespace(root=str(root)))
    args = parser.parse_args([control.command, "start"])
    args.path = str(root)
    package = Mock()
    importer = Mock(return_value=package)
    monkeypatch.setattr(
        "fedbiomed.node.cli.importlib", SimpleNamespace(import_module=importer)
    )
    proc = Mock(pid=12345)
    proc.wait.return_value = 0
    proc.__enter__ = Mock(return_value=proc)
    proc.__exit__ = Mock(return_value=False)
    popen = Mock(return_value=proc)
    monkeypatch.setattr("fedbiomed.node.cli.subprocess.Popen", popen)
    return SimpleNamespace(
        control=control,
        parser=parser,
        args=args,
        package=package,
        importer=importer,
        proc=proc,
        popen=popen,
        data=data,
    )


def test_production_uses_active_python_and_does_not_mutate_environment(launcher):
    before = os.environ.copy()
    launcher.control.forward(launcher.args, [])
    command = launcher.popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "gunicorn"]
    assert command[-1] == launcher.control.application
    assert launcher.popen.call_args.kwargs["start_new_session"] is True
    assert "shell" not in launcher.popen.call_args.kwargs
    env = launcher.popen.call_args.kwargs["env"]
    assert env["DATA_PATH"] == str(launcher.data)
    assert env["FBM_NODE_COMPONENT_ROOT"] == launcher.args.path
    assert os.environ == before
    launcher.importer.assert_called_once_with(launcher.control.module)
    launcher.package.get_frontend_source_dir.assert_not_called()


@pytest.mark.parametrize("development", [False, True])
def test_server_specific_tls_and_debug_flags(launcher, development):
    launcher.args.development = development
    launcher.args.debug = development
    launcher.args.cert_file = "/certificates/server cert.pem"
    launcher.args.key_file = "/certificates/server key.pem"
    launcher.control.forward(launcher.args, [])
    command = launcher.popen.call_args.args[0]
    if development:
        assert command[:3] == [sys.executable, "-m", "flask"]
        assert command[command.index("--app") + 1] == launcher.control.application
        assert "--debug" in command
        assert "--certfile" not in command
        assert command[command.index("--cert") + 1] == launcher.args.cert_file
        assert command[command.index("--key") + 1] == launcher.args.key_file
    else:
        assert command[command.index("--certfile") + 1] == launcher.args.cert_file
        assert command[command.index("--keyfile") + 1] == launcher.args.key_file


def test_development_does_not_inherit_debug(launcher, monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", "1")
    launcher.args.development = True
    launcher.control.forward(launcher.args, [])
    assert "--no-debug" in launcher.popen.call_args.args[0]


@pytest.mark.parametrize(
    "option,message",
    [
        ("key_file", "both --cert-file and --key-file"),
        ("cert_file", "both --cert-file and --key-file"),
        ("debug", "--debug requires --development"),
        ("data_folder", "is not a folder"),
    ],
)
def test_invalid_options_fail_before_start(launcher, option, message, capsys):
    setattr(launcher.args, option, True if option == "debug" else "/missing")
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value.code == 1
    assert message in capsys.readouterr().out
    launcher.popen.assert_not_called()


def test_missing_optional_package_has_versioned_install_hint(
    launcher, monkeypatch, capsys
):
    monkeypatch.setattr("fedbiomed.node.cli.__version__", "v7.2.1rc1")
    launcher.importer.side_effect = ModuleNotFoundError(name=launcher.control.module)
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert f"fedbiomed[{launcher.control.extra}]==7.2.1rc1" in output
    assert launcher.control.module.replace("_", "-") in output
    assert "Traceback" not in output
    launcher.popen.assert_not_called()


def test_transitive_import_failure_is_not_misreported(launcher):
    error = ModuleNotFoundError(
        "No module named 'broken_dependency'", name="broken_dependency"
    )
    launcher.importer.side_effect = error
    with pytest.raises(ModuleNotFoundError) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value is error
    launcher.popen.assert_not_called()


def test_unknown_arguments_are_rejected(launcher):
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, ["--typo"])
    assert exc.value.code == 2
    launcher.importer.assert_not_called()
    launcher.popen.assert_not_called()


@pytest.mark.parametrize("returncode,expected", [(3, 3), (-signal.SIGTERM, 143)])
def test_server_failure_reaches_cli_exit_code(launcher, returncode, expected):
    launcher.proc.wait.return_value = returncode
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value.code == expected


def test_process_start_error_is_reported(launcher, capsys):
    launcher.popen.side_effect = OSError("server executable unavailable")
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value.code == 1
    assert "server executable unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("timeout", [False, True])
def test_signals_stop_the_server_group_and_restore_handlers(
    launcher, monkeypatch, signum, timeout
):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    killpg = Mock()
    monkeypatch.setattr("fedbiomed.node.cli.os.killpg", killpg)

    def interrupt():
        signal.getsignal(signum)(signum, None)

    results = [interrupt]
    if timeout:
        results.append(subprocess.TimeoutExpired("server", 10))
    results.append(0)

    def wait(**kwargs):
        result = results.pop(0)
        if callable(result):
            return result()
        if isinstance(result, Exception):
            raise result
        return result

    launcher.proc.wait.side_effect = wait
    with pytest.raises(SystemExit) as exc:
        launcher.control.forward(launcher.args, [])
    assert exc.value.code == 128 + signum
    assert killpg.call_args_list[0].args == (launcher.proc.pid, signum)
    if timeout:
        assert killpg.call_args_list[1].args == (launcher.proc.pid, signal.SIGKILL)
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_node_registers_api_without_gui_options():
    cli = NodeCLI()
    args = cli.parser.parse_args(["api", "start"])
    assert isinstance(args.func.__self__, APIControl)
    assert not hasattr(args, "recreate")
    with pytest.raises(SystemExit) as exc:
        cli.parser.parse_args(["api", "start", "--recreate"])
    assert exc.value.code == 2
