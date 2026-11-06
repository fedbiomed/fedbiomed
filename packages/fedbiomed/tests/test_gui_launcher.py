"""Check the core CLI's GUI launcher without importing the GUI package."""

import argparse
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import GUIControl


class TestGUIControl(unittest.TestCase):
    """Tests for GUIControl argument parser."""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = GUIControl(self.subparsers)
        self.context = MagicMock()
        self.control._context = self.context

    def test_01_gui_control_initialize(self):
        """Tests gui subparser and all its start options are registered."""
        self.control.initialize()
        self.assertIn("gui", self.subparsers.choices)

        gui_subparsers = (
            self.subparsers.choices["gui"]._subparsers._group_actions[0].choices
        )
        self.assertIn("start", gui_subparsers)

        start_options = gui_subparsers["start"]._option_string_actions
        for opt in [
            "--port",
            "--host",
            "--cert-file",
            "--key-file",
            "--debug",
            "--recreate",
            "--development",
            "--data-folder",
        ]:
            self.assertIn(opt, start_options)

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_02_gui_control_forward_gunicorn(
        self, mock_isdir, mock_importlib, mock_subprocess
    ):
        """Tests forward() launches gunicorn when development mode is off."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        args = argparse.Namespace(
            path="/some/fedbiomed/path",
            data_folder="/test/data",
            key_file=None,
            cert_file=None,
            development=False,
            host="localhost",
            port="8484",
            debug=False,
            recreate=False,
        )
        self.control.forward(args, [])

        mock_subprocess.Popen.assert_called_once()
        command = mock_subprocess.Popen.call_args[0][0]
        env = mock_subprocess.Popen.call_args[1]["env"]
        self.assertIn("gunicorn", command)
        self.assertEqual(env["DATA_PATH"], "/test/data")
        self.assertEqual(env["FBM_NODE_COMPONENT_ROOT"], "/some/fedbiomed/path")
        self.assertNotIn("FBM_START_NODE_WITH_GUI", env)
        self.assertNotIn("FBM_NODE_START_ARGS", env)

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_03_gui_control_forward_development_mode(
        self, mock_isdir, mock_importlib, mock_subprocess
    ):
        """Tests forward() uses flask when development mode is on."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        args = argparse.Namespace(
            path="/some/fedbiomed/path",
            data_folder="/test/data",
            key_file=None,
            cert_file=None,
            development=True,
            host="localhost",
            port="8484",
            debug=False,
            recreate=False,
        )
        self.control.forward(args, [])

        command = mock_subprocess.Popen.call_args[0][0]
        self.assertIn("flask", command)

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_04_gui_control_forward_ssl_certificates(
        self, mock_isdir, mock_importlib, mock_subprocess
    ):
        """Tests forward() passes --keyfile/--certfile to gunicorn when SSL is configured."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        args = argparse.Namespace(
            path="/some/fedbiomed/path",
            data_folder="/test/data",
            key_file="server.key",
            cert_file="server.pem",
            development=False,
            host="localhost",
            port="8484",
            debug=False,
            recreate=False,
        )
        self.control.forward(args, [])

        command = mock_subprocess.Popen.call_args[0][0]
        self.assertIn("--keyfile", command)
        self.assertIn("--certfile", command)

    @patch("os.path.isdir", return_value=False)
    def test_05_gui_control_forward_invalid_data_folder(self, mock_isdir):
        """Tests forward() raises FedbiomedError when the data folder does not exist."""
        self.control.initialize()
        self.context.config.root = "/node/root"

        args = argparse.Namespace(
            path="/some/fedbiomed/path",
            data_folder="/nonexistent/data",
            key_file=None,
            cert_file=None,
            development=False,
            host="localhost",
            port="8484",
            debug=False,
            recreate=False,
        )
        with self.assertRaises(FedbiomedError):
            self.control.forward(args, [])

    def _recreate_args(self):
        return argparse.Namespace(
            path="/some/fedbiomed/path",
            data_folder="/test/data",
            key_file=None,
            cert_file=None,
            development=False,
            host="localhost",
            port="8484",
            debug=False,
            recreate=True,
        )

    @patch("fedbiomed.node.cli.shutil.which", return_value="/usr/bin/yarn")
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_06_gui_control_forward_recreate_builds_before_launching(
        self, mock_isdir, mock_importlib, mock_subprocess, mock_is_file, mock_which
    ):
        """Tests --recreate installs and builds the ui, then launches the server."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        self.control.forward(self._recreate_args(), [])

        ui_folder = Path("/path/to/fedbiomed_gui/ui")
        self.assertEqual(
            mock_subprocess.run.call_args_list,
            [
                call(["/usr/bin/yarn", "install"], cwd=ui_folder, check=True),
                call(["/usr/bin/yarn", "build"], cwd=ui_folder, check=True),
            ],
        )
        mock_subprocess.Popen.assert_called_once()

    @patch("pathlib.Path.is_file", return_value=False)
    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_07_gui_control_forward_recreate_without_sources(
        self, mock_isdir, mock_importlib, mock_subprocess, mock_is_file
    ):
        """Tests --recreate raises, and launches nothing, without the ui sources."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        with self.assertRaises(FedbiomedError):
            self.control.forward(self._recreate_args(), [])

        mock_subprocess.run.assert_not_called()
        mock_subprocess.Popen.assert_not_called()

    @patch("fedbiomed.node.cli.shutil.which", return_value=None)
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_08_gui_control_forward_recreate_without_yarn(
        self, mock_isdir, mock_importlib, mock_subprocess, mock_is_file, mock_which
    ):
        """Tests --recreate raises, and launches nothing, when yarn is missing."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )

        with self.assertRaises(FedbiomedError):
            self.control.forward(self._recreate_args(), [])

        mock_subprocess.Popen.assert_not_called()

    @patch("fedbiomed.node.cli.shutil.which", return_value="/usr/bin/yarn")
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_09_gui_control_forward_recreate_build_fails(
        self, mock_isdir, mock_importlib, mock_subprocess, mock_is_file, mock_which
    ):
        """Tests a failed build raises, and launches nothing."""
        self.control.initialize()
        self.context.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        mock_subprocess.run.side_effect = [
            None,
            subprocess.CalledProcessError(1, ["/usr/bin/yarn", "build"]),
        ]

        with self.assertRaisesRegex(FedbiomedError, "yarn build"):
            self.control.forward(self._recreate_args(), [])

        mock_subprocess.Popen.assert_not_called()
