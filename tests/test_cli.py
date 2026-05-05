import argparse
import configparser
import io
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fedbiomed
import fedbiomed.node.cli_utils
from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import (
    DatasetArgumentParser,
    GUIControl,
    NodeCLI,
    NodeControl,
    TrainingPlanArgumentParser,
    _find_available_port,
    _node_signal_trigger_term,
    intro,
    start_node,
)
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
    get_map_modalities2folders_from_cli,
)
from fedbiomed.node.config import NodeConfig

# ============================================================================
# SHARED FIXTURES AND HELPERS
# ============================================================================


@pytest.fixture
def temp_medical_data():
    """Fixture to create temporary test data."""
    temp_dir = tempfile.mkdtemp()
    test_path = Path(temp_dir) / "medical_data"
    test_path.mkdir()

    # Create a mock CSV file path
    csv_path = Path(temp_dir) / "demographics.csv"
    csv_path.touch()

    yield temp_dir, str(test_path), str(csv_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_stdio():
    """Fixture to capture stdout/stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    yield

    sys.stdout = old_stdout
    sys.stderr = old_stderr


@pytest.fixture
def mock_medical_folder_controller():
    """Fixture for common MedicalFolderController mock setup."""
    mock_controller = MagicMock()
    mock_controller.df_dir = {"modality": MagicMock()}
    mock_controller.df_dir["modality"].unique.return_value = ["T1", "T2", "label"]
    mock_controller.demographics_column_names.return_value = [
        "subject_id",
        "age",
        "gender",
    ]
    return mock_controller


class MockInputHelper:
    """Helper class to manage mock inputs for get_map_modalities2folders_from_cli tests."""

    inputs = []

    @staticmethod
    def mock_input(x):
        return MockInputHelper.inputs.pop(0)


@pytest.fixture
def node_control_parser():
    """Fixture for NodeControl parser tests."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    control = NodeControl(subparsers)
    node = MagicMock()
    control._node = node
    return parser, subparsers, control, node


@pytest.fixture
def gui_control_parser():
    """Fixture for GUIControl parser tests."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    control = GUIControl(subparsers)
    node = MagicMock()
    control._node = node
    return parser, subparsers, control, node


# ============================================================================
# UNITTEST TESTCASES (Legacy - Consider converting to pytest)
# ============================================================================


class TestTrainingPlanArgumentParser(unittest.TestCase):
    """Test case for node cli dataset argument parse"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.tp_arg_pars = TrainingPlanArgumentParser(self.subparsers)
        self.tp_arg_pars._node = MagicMock()

    def test_01_training_plan_argument_parser_initialize(self):
        """Tests training plan parser initialization"""
        self.tp_arg_pars.initialize()
        self.assertTrue("training-plan" in self.subparsers.choices)

        tp_choices = (
            self.subparsers.choices["training-plan"]
            ._subparsers._group_actions[0]
            .choices
        )

        self.assertTrue("approve" in tp_choices)  # noqa
        self.assertTrue("list" in tp_choices)  # noqa
        self.assertTrue("delete" in tp_choices)  # noqa
        self.assertTrue("reject" in tp_choices)  # noqa
        self.assertTrue("view" in tp_choices)  # noqa
        self.assertTrue("update" in tp_choices)  # noqa

    def test_02_training_plan_argument_parser_execute(self):
        """Tests training plan argument parser actions"""

        self.tp_arg_pars.initialize()

        args = self.parser.parse_args(["training-plan", "delete"])
        self.tp_arg_pars.delete(args)
        self.tp_arg_pars.list()
        self.tp_arg_pars.update()

        with patch.object(fedbiomed.node.cli, "view_training_plan") as m:
            self.tp_arg_pars.view()
            m.assert_called_once()

        with patch.object(fedbiomed.node.cli, "register_training_plan") as m:
            self.tp_arg_pars.register()
            m.assert_called_once()

        with patch.object(fedbiomed.node.cli, "reject_training_plan") as m:
            args = self.parser.parse_args(["training-plan", "reject"])
            self.tp_arg_pars.reject(args)
            m.assert_called_once()

    def test_03_training_plan_argument_parser_approve(self):
        """Tests training plan approve action calls approve_training_plan."""
        self.tp_arg_pars.initialize()
        with patch.object(fedbiomed.node.cli, "approve_training_plan") as m:
            args = self.parser.parse_args(["training-plan", "approve"])
            self.tp_arg_pars.approve(args)
            m.assert_called_once()


class TestDatasetArgumentParser(unittest.TestCase):
    """Test case for node cli dataset argument parse"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.dataset_arg_pars = DatasetArgumentParser(self.subparsers)
        self.node = MagicMock()
        self.dataset_arg_pars._node = self.node

    def test_01_dataset_argument_parser_initialize(self):
        """Test initialization"""

        self.dataset_arg_pars.initialize()
        self.assertTrue("dataset" in self.subparsers.choices)
        dataset_choices = (
            self.subparsers.choices["dataset"]._subparsers._group_actions[0].choices
        )

        self.assertTrue("add" in dataset_choices)  # noqa
        self.assertTrue("list" in dataset_choices)  # noqa
        self.assertTrue("delete" in dataset_choices)  # noqa

        self.assertTrue("--mnist" in dataset_choices["add"]._option_string_actions)
        self.assertTrue("--file" in dataset_choices["add"]._option_string_actions)

        self.assertTrue("--all" in dataset_choices["delete"]._option_string_actions)
        self.assertTrue("--mnist" in dataset_choices["delete"]._option_string_actions)

    def test_02_dataset_argument_parser_add(self):
        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "add"])

        with patch.object(fedbiomed.node.cli, "add_database") as m:
            self.dataset_arg_pars.add(args)
            m.assert_called_once()
            m.reset_mock()

            args = self.parser.parse_args(["dataset", "add", "--mnist", "test"])
            self.dataset_arg_pars.add(args)
            m.assert_called_once_with(
                self.node.dataset_manager, interactive=False, path="test"
            )
            m.reset_mock()

    def test_03_dataset_argument_parser_delete(self):
        self.dataset_arg_pars.initialize()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager)

        with patch.object(fedbiomed.node.cli, "delete_all_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--all"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager)
            m.reset_mock()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--mnist"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.node.dataset_manager, interactive=False)

    def test_04_dataset_argument_parser_list(self):
        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "list"])
        self.dataset_arg_pars.list(args)

        self.node.dataset_manager.list_my_datasets.assert_called_once_with(verbose=True)

    def test_05_dataset_argument_parser_add_file(self):
        """Tests add() dispatches to _add_dataset_from_file when --file is given."""
        self.dataset_arg_pars.initialize()
        with patch.object(self.dataset_arg_pars, "_add_dataset_from_file") as m:
            args = self.parser.parse_args(
                ["dataset", "add", "--file", "/path/to/dataset.json"]
            )
            self.dataset_arg_pars.add(args)
            m.assert_called_once_with(path="/path/to/dataset.json")

    def test_06_dataset_argument_parser_add_mnist_default_path(self):
        """Tests add() uses node data folder when --mnist is given without a value."""
        self.dataset_arg_pars.initialize()
        self.node.config.root = "/test/root"
        with patch.object(fedbiomed.node.cli, "add_database") as m:
            args = self.parser.parse_args(["dataset", "add", "--mnist"])
            self.dataset_arg_pars.add(args)
            expected_path = os.path.join("/test/root", NODE_DATA_FOLDER)
            m.assert_called_once_with(
                self.node.dataset_manager, interactive=False, path=expected_path
            )

    def test_07_add_dataset_from_file_absolute_path(self):
        """Tests _add_dataset_from_file preserves absolute paths."""
        self.dataset_arg_pars.initialize()
        dataset_json = {
            "path": "/absolute/data/path",
            "data_type": "csv",
            "description": "Test dataset",
            "tags": "#test",
            "name": "TestData",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset_json, f)
            json_path = f.name
        try:
            with patch.object(fedbiomed.node.cli, "add_database") as m:
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                m.assert_called_once_with(
                    self.node.dataset_manager,
                    interactive=False,
                    path="/absolute/data/path",
                    data_type="csv",
                    description="Test dataset",
                    tags="#test",
                    name="TestData",
                    dataset_parameters=None,
                )
        finally:
            os.unlink(json_path)

    def test_08_add_dataset_from_file_relative_path(self):
        """Tests _add_dataset_from_file prepends config.root for relative paths."""
        self.dataset_arg_pars.initialize()
        self.node.config.root = "/test/root"
        dataset_json = {
            "path": "relative/data",
            "data_type": "csv",
            "description": "Test",
            "tags": "#test",
            "name": "RelData",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset_json, f)
            json_path = f.name
        try:
            with patch.object(fedbiomed.node.cli, "add_database") as m:
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                call_kwargs = m.call_args[1]
                self.assertIn("/test/root", call_kwargs["path"])
                self.assertIn("relative", call_kwargs["path"])
        finally:
            os.unlink(json_path)

    def test_09_add_dataset_from_file_env_var_path(self):
        """Tests _add_dataset_from_file expands environment variable in path."""
        self.dataset_arg_pars.initialize()
        dataset_json = {
            "path": "$FBM_TEST_DATA/data",
            "data_type": "csv",
            "description": "Test",
            "tags": "#test",
            "name": "EnvData",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset_json, f)
            json_path = f.name
        try:
            with patch.dict(os.environ, {"FBM_TEST_DATA": "/env/path"}):
                with patch.object(fedbiomed.node.cli, "add_database") as m:
                    self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                    call_kwargs = m.call_args[1]
                    self.assertIn("/env/path", call_kwargs["path"])
        finally:
            os.unlink(json_path)

    def test_10_add_dataset_from_file_invalid_json(self):
        """Tests _add_dataset_from_file exits when the file is not valid JSON."""
        self.dataset_arg_pars.initialize()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            json_path = f.name
        try:
            with self.assertRaises(SystemExit):
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)
        finally:
            os.unlink(json_path)


class TestNodeControl(unittest.TestCase):
    """Test case for node control parser"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = NodeControl(self.subparsers)
        self.node = MagicMock()
        self.control._node = self.node

    def test_01_node_control_initialize(self):
        """Tests initialize"""

        self.control.initialize()
        self.assertTrue("start" in self.subparsers.choices)
        self.assertTrue(
            "--gpu-only" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa
        self.assertTrue(
            "--gpu-num" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa
        self.assertTrue(
            "--gpu" in self.subparsers.choices["start"]._option_string_actions
        )  # noqa

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("fedbiomed.node.cli._start_managed_node")
    @patch("fedbiomed.node.cli._missing_restful_gui_dependencies", return_value=[])
    @patch("os.path.isdir", return_value=True)
    def test_02_node_control_start(
        self,
        mock_isdir,
        mock_missing_dependencies,
        mock_start_managed_node,
        mock_importlib,
        mock_subprocess,
    ):
        self.control.initialize()
        self.node.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )
        args = self.parser.parse_args(["start"])
        os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"

        self.control.start(args)
        mock_subprocess.Popen.assert_called_once()
        command = mock_subprocess.Popen.call_args[0][0]
        env = mock_subprocess.Popen.call_args[1]["env"]
        self.assertIn("gunicorn", command)
        self.assertEqual(env["FBM_NODE_COMPONENT_ROOT"], "/node/root")
        self.assertEqual(env["FBM_START_NODE_WITH_RESTFUL"], "true")
        self.assertEqual(env["FBM_RESTFUL_HOST"], "localhost")
        self.assertEqual(env["FBM_RESTFUL_PORT"], "8484")
        self.assertEqual(
            env["FBM_NODE_START_ARGS"],
            json.dumps({"gpu": False, "gpu_num": 1, "gpu_only": False, "debug": False}),
        )
        mock_missing_dependencies.assert_called_once_with(False)
        mock_start_managed_node.assert_not_called()

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("fedbiomed.node.cli._missing_restful_gui_dependencies", return_value=[])
    @patch("os.path.isdir", return_value=True)
    def test_04_node_control_start_gpu_and_debug_flags(
        self,
        mock_isdir,
        mock_missing_dependencies,
        mock_importlib,
        mock_subprocess,
    ):
        """Tests GPU and debug flags are correctly forwarded in node startup args."""
        self.control.initialize()
        self.node.config.root = "/node/root"
        mock_importlib.import_module.return_value.__file__ = (
            "/path/to/fedbiomed_gui/__init__.py"
        )
        os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"

        args = self.parser.parse_args(
            ["start", "--gpu", "--gpu-num", "2", "--gpu-only", "--debug"]
        )
        self.control.start(args)

        env = mock_subprocess.Popen.call_args[1]["env"]
        node_args = json.loads(env["FBM_NODE_START_ARGS"])
        self.assertTrue(node_args["gpu"])
        self.assertEqual(node_args["gpu_num"], 2)
        self.assertTrue(node_args["gpu_only"])
        self.assertTrue(node_args["debug"])

    @patch("fedbiomed.node.cli.Node", autospec=True)
    def test_03_node_control__start(self, mock_node):
        """Tests node start"""

        cfg = configparser.ConfigParser()
        cfg["security"] = {
            "training_plan_apprival": "true",
            "allow_default_training_plan": "true",
        }
        cfg["default"] = {"id": "test-id"}

        mock_node.return_value.tp_security_manager = MagicMock()

        with tempfile.TemporaryDirectory() as temp_:
            config = NodeConfig(temp_)
            config._cfg = cfg
            args = {"gpu": False}
            config._cfg["security"]["training_plan_approval"] = "false"
            start_node("config.ini", args)
            mock_node.return_value.task_manager.assert_called_once()

            with patch.object(fedbiomed.node.cli, "logger") as logger:
                mock_node.return_value.task_manager.side_effect = FedbiomedError
                start_node("config.ini", args)
                logger.critical.assert_called_once()
                logger.critical.reset_mock()

                mock_node.return_value.task_manager.side_effect = Exception
                start_node("config.ini", args)
                logger.critical.assert_called_once()


class TestGUIControl(unittest.TestCase):
    """Tests for GUIControl argument parser."""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = GUIControl(self.subparsers)
        self.node = MagicMock()
        self.control._node = self.node

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
        self.node.config.root = "/node/root"
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
        self.assertEqual(env["FBM_START_NODE_WITH_RESTFUL"], "true")
        self.assertEqual(
            env["FBM_NODE_START_ARGS"],
            json.dumps({"gpu": False, "gpu_num": 1, "gpu_only": False, "debug": False}),
        )
        self.assertEqual(env["FBM_RESTFUL_HOST"], "localhost")
        self.assertEqual(env["FBM_RESTFUL_PORT"], "8484")

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_03_gui_control_forward_development_mode(
        self, mock_isdir, mock_importlib, mock_subprocess
    ):
        """Tests forward() uses flask when development mode is on."""
        self.control.initialize()
        self.node.config.root = "/node/root"
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
        env = mock_subprocess.Popen.call_args[1]["env"]
        self.assertIn("flask", command)
        self.assertEqual(env["FBM_DEBUG"], "false")

    @patch("fedbiomed.node.cli.subprocess")
    @patch("fedbiomed.node.cli.importlib")
    @patch("os.path.isdir", return_value=True)
    def test_04_gui_control_forward_ssl_certificates(
        self, mock_isdir, mock_importlib, mock_subprocess
    ):
        """Tests forward() passes --keyfile/--certfile to gunicorn when SSL is configured."""
        self.control.initialize()
        self.node.config.root = "/node/root"
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
        self.node.config.root = "/node/root"

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


class TestStartNode(unittest.TestCase):
    """Tests for the start_node function."""

    @patch("fedbiomed.node.cli.Node")
    def test_01_start_node_training_plan_approval_with_default_plans(self, mock_node):
        """Tests tp_security_manager methods are called when approval + default plans are enabled."""
        mock_node.return_value.config.getbool.return_value = True

        start_node("config.ini", {"gpu": False})

        mock_node.return_value.tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
        mock_node.return_value.tp_security_manager.register_update_default_training_plans.assert_called_once()

    @patch("fedbiomed.node.cli.Node")
    def test_02_start_node_training_plan_approval_no_default_plans(self, mock_node):
        """Tests register_update_default_training_plans is NOT called when allow_default_training_plans is False."""

        def _getbool(section, key):
            return key == "training_plan_approval"

        mock_node.return_value.config.getbool.side_effect = _getbool

        start_node("config.ini", {"gpu": False})

        mock_node.return_value.tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
        mock_node.return_value.tp_security_manager.register_update_default_training_plans.assert_not_called()


def test_node_control_start_falls_back_when_restful_gui_deps_missing(
    node_control_parser, mocker, monkeypatch
):
    """Tests node start falls back to federation node when REST/GUI deps are missing."""
    parser, _, control, node = node_control_parser
    control.initialize()
    node.config.root = "/node/root"
    mocker.patch(
        "fedbiomed.node.cli._missing_restful_gui_dependencies",
        return_value=["flask", "gunicorn"],
    )
    mock_start_managed_node = mocker.patch("fedbiomed.node.cli._start_managed_node")
    monkeypatch.setenv("FEDBIOMED_ACTIVE_NODE_ID", "test-node-id")

    args = parser.parse_args(["start"])
    control.start(args)

    mock_start_managed_node.assert_called_once_with(
        node.config,
        {"gpu": False, "gpu_num": 1, "gpu_only": False, "debug": False},
    )


def test_node_control_start_no_gui_starts_managed_node_only(
    node_control_parser, mocker, monkeypatch
):
    """Tests --no-gui bypasses REST/GUI startup and starts only the node."""
    parser, _, control, node = node_control_parser
    control.initialize()
    node.config.root = "/node/root"
    mock_start_managed_node = mocker.patch("fedbiomed.node.cli._start_managed_node")
    monkeypatch.setenv("FEDBIOMED_ACTIVE_NODE_ID", "test-node-id")

    args = parser.parse_args(["start", "--no-gui", "--debug"])
    control.start(args)

    mock_start_managed_node.assert_called_once_with(
        node.config,
        {"gpu": False, "gpu_num": 1, "gpu_only": False, "debug": True},
    )


def test_gui_control_forward_uses_next_available_port(gui_control_parser, mocker):
    """Tests forward() uses the selected available port for command and env."""
    _, _, control, node = gui_control_parser
    control.initialize()
    node.config.root = "/node/root"
    mocker.patch("os.path.isdir", return_value=True)
    mock_find_available_port = mocker.patch(
        "fedbiomed.node.cli._find_available_port", return_value="8485"
    )
    mock_importlib = mocker.patch("fedbiomed.node.cli.importlib")
    mock_subprocess = mocker.patch("fedbiomed.node.cli.subprocess")
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
    control.forward(args, [])

    command = mock_subprocess.Popen.call_args[0][0]
    env = mock_subprocess.Popen.call_args[1]["env"]
    mock_find_available_port.assert_called_once_with("localhost", "8484")
    assert "localhost:8485" in command
    assert env["FBM_RESTFUL_PORT"] == "8485"


def test_find_available_port_skips_occupied_port():
    """Tests occupied ports are skipped in favor of a later available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        occupied_port = sock.getsockname()[1]

        available_port = int(_find_available_port("localhost", str(occupied_port)))

    assert available_port > occupied_port


def test_find_available_port_rejects_invalid_port():
    """Tests invalid port values raise a FedbiomedError."""
    with pytest.raises(FedbiomedError):
        _find_available_port("localhost", "not-a-port")


class TestNodeCLI(unittest.TestCase):
    """Tests main NodeCLI"""

    @patch("builtins.input")
    def test_01_node_cli_init(self, input_patch):
        """Tests initialization"""
        input_patch.return_value = "y"
        # import sys
        # sys.argv.append('-y')
        # remove any `fbm-node` folder already existing
        shutil.rmtree("fbm-node", ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.node_cli = NodeCLI()
            self.node_cli.parse_args(
                ["--path", os.path.join(str(tmpdirname), "fbm-node"), "dataset", "list"]
            )
        # sys.argv.remove('-y')


# ============================================================================
# PYTEST TESTS FOR MEDICAL FOLDER DATASET CLI UTILITIES
# ============================================================================


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
def test_add_medical_folder_dataset_no_demographics(
    mock_input,
    mock_controller_class,
    mock_validated_path,
    temp_medical_data,
    mock_medical_folder_controller,
):
    """Test adding medical folder dataset without demographics file."""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.return_value = test_path
    mock_input.return_value = "n"  # No demographics file
    mock_controller_class.return_value = mock_medical_folder_controller

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=None, dlp=None
    )

    # Assertions
    assert result_path == test_path
    assert result_params == {}
    assert result_dlp is None

    # Verify mocks were called correctly
    mock_validated_path.assert_called_once_with(type="dir")
    mock_controller_class.assert_called_once_with(test_path)
    mock_input.assert_called_once_with(
        "\nWould you like to select a demographics csv file? [y/N]\n"
    )


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
def test_add_medical_folder_dataset_demographics(
    mock_input,
    mock_controller_class,
    mock_validated_path,
    temp_medical_data,
    mock_medical_folder_controller,
):
    """Test adding medical folder dataset with demographics file"""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.side_effect = [test_path, csv_path]
    mock_input.side_effect = ["y", "1"]  # Yes to demographics, index column 1
    mock_controller_class.return_value = mock_medical_folder_controller

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters={"existing_param": "value"}, dlp=None
    )

    # Assertions
    assert result_path == test_path
    assert result_params == {
        "existing_param": "value",
        "tabular_file": csv_path,
        "index_col": 1,
    }
    assert result_dlp is None

    # Verify mocks were called correctly
    assert mock_validated_path.call_count == 2
    mock_medical_folder_controller.demographics_column_names.assert_called_once_with(
        csv_path
    )


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
@patch("warnings.warn")
def test_add_medical_folder_dataset_invalid_index_then_valid(
    mock_warn, mock_input, mock_controller_class, mock_validated_path, temp_medical_data
):
    """Test handling invalid index input followed by valid input."""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.side_effect = [test_path, csv_path]
    mock_input.side_effect = [
        "y",
        "invalid",
        "2",
    ]  # Yes, invalid input, then valid index

    mock_controller = MagicMock()
    mock_controller.df_dir = {"modality": MagicMock()}
    mock_controller.df_dir["modality"].unique.return_value = ["T1"]
    mock_controller.demographics_column_names.return_value = ["id", "age", "status"]
    mock_controller_class.return_value = mock_controller

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=None, dlp=None
    )

    # Assertions
    assert result_params["index_col"] == 2
    mock_warn.assert_called_once_with(
        "Please input a numeric value (integer)", stacklevel=1
    )


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
def test_add_medical_folder_dataset_with_existing_dlp(
    mock_input, mock_controller_class, mock_validated_path, temp_medical_data
):
    """Test function with existing DataLoadingPlan."""
    from fedbiomed.common.dataloadingplan import DataLoadingPlan

    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.return_value = test_path
    mock_input.return_value = "n"  # No demographics

    mock_controller = MagicMock()
    mock_controller.df_dir = {"modality": MagicMock()}
    mock_controller.df_dir["modality"].unique.return_value = ["T1"]
    mock_controller_class.return_value = mock_controller

    # Create existing DLP
    existing_dlp = DataLoadingPlan()

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=None, dlp=existing_dlp
    )

    # Assertions
    assert result_dlp == existing_dlp


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
def test_add_medical_folder_dataset_preserves_existing_parameters(
    mock_input, mock_controller_class, mock_validated_path, temp_medical_data
):
    """Test that existing dataset parameters are preserved."""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.return_value = test_path
    mock_input.return_value = "n"  # No demographics

    mock_controller = MagicMock()
    mock_controller.df_dir = {"modality": MagicMock()}
    mock_controller.df_dir["modality"].unique.return_value = ["T1", "T2"]
    mock_controller_class.return_value = mock_controller

    existing_params = {"custom_param": "value", "another_param": 123}

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=existing_params, dlp=None
    )

    # Assertions
    assert result_params["custom_param"] == "value"
    assert result_params["another_param"] == 123
    assert len(result_params) == 2  # No additional params added


# Test scenarios for modality mapping with parametrization
@pytest.mark.parametrize(
    "scenario,inputs,expected_map,expect_warnings",
    [
        ("basic_mapping", ["1"], {"T1": ["Should map to T1"]}, False),
        (
            "wrong_inputs_then_correct",
            ["wrong", "5", "1"],
            {"T1": ["Should map to T1"]},
            True,
        ),
        (
            "add_new_modality",
            ["1", "0", "Tnew", "y", "4"],
            {"T1": ["Should map to T1"], "Tnew": ["Should map to Tnew"]},
            False,
        ),
        (
            "complex_scenario",
            ["1", "0", "Tmistake", "n", "Tnew", "", "4", "5", "1", "2"],
            {
                "T1": ["Should map to T1", "Should also map to T1"],
                "Tnew": ["Should map to Tnew"],
                "T2": ["Should map to T2"],
            },
            True,
        ),
    ],
)
@patch(
    "fedbiomed.node.cli_utils._medical_folder_dataset.input",
    new=MockInputHelper.mock_input,
)
def test_get_map_modalities2folders_from_cli_scenarios(
    mock_stdio, scenario, inputs, expected_map, expect_warnings
):
    """Test various scenarios for modality to folder mapping."""
    if expect_warnings:
        pytest.mark.filterwarnings("ignore::UserWarning")

    # Set up test data based on scenario
    if scenario == "basic_mapping" or scenario == "wrong_inputs_then_correct":
        modality_folder_names = ["Should map to T1"]
    elif scenario == "add_new_modality":
        modality_folder_names = ["Should map to T1", "Should map to Tnew"]
    else:  # complex_scenario
        modality_folder_names = [
            "Should map to T1",
            "Should map to Tnew",
            "Should also map to T1",
            "Should map to T2",
        ]

    MockInputHelper.inputs = inputs.copy()

    if expect_warnings:
        with pytest.warns(UserWarning):
            dlb = get_map_modalities2folders_from_cli(modality_folder_names)
    else:
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)

    assert dlb.map == expected_map


def test_intro():
    """Tests intro() prints the active node ID from the environment."""
    os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"
    with patch("builtins.print") as mock_print:
        intro()
    printed = " ".join(
        str(arg) for call in mock_print.call_args_list for arg in call.args
    )
    assert "test-node-id" in printed


def test_node_signal_trigger_term():
    """Tests _node_signal_trigger_term sends SIGTERM to the current process."""
    with patch("os.kill") as mock_kill:
        _node_signal_trigger_term()
    mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
