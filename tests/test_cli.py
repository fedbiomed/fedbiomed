import argparse
import io
import json
import os
import signal
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

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
    intro,
)
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
    get_map_modalities2folders_from_cli,
)
from fedbiomed.node.node import NodeContext
from fedbiomed.node.node_pm import _node_signal_trigger_term, _start_node_process

# ============================================================================
# SHARED FIXTURES AND HELPERS
# ============================================================================


@pytest.fixture
def temp_medical_data():
    """Fixture to provide fake test data paths."""
    return "/fake/tmp", "/fake/medical_data", "/fake/demographics.csv"


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


# ============================================================================
# UNITTEST TESTCASES (Legacy - Consider converting to pytest)
# ============================================================================


class TestTrainingPlanArgumentParser(unittest.TestCase):
    """Test case for node cli dataset argument parse"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.tp_arg_pars = TrainingPlanArgumentParser(self.subparsers)
        self.tp_arg_pars._context = MagicMock()

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
        self.context = MagicMock()
        self.dataset_arg_pars._context = self.context

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
                self.context.dataset_manager, interactive=False, path="test"
            )
            m.reset_mock()

    def test_03_dataset_argument_parser_delete(self):
        self.dataset_arg_pars.initialize()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.context.dataset_manager)

        with patch.object(fedbiomed.node.cli, "delete_all_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--all"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.context.dataset_manager)
            m.reset_mock()

        with patch.object(fedbiomed.node.cli, "delete_database") as m:
            args = self.parser.parse_args(["dataset", "delete", "--mnist"])
            self.dataset_arg_pars.delete(args)
            m.assert_called_once_with(self.context.dataset_manager, interactive=False)

    def test_04_dataset_argument_parser_list(self):
        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "list"])
        self.dataset_arg_pars.list(args)

        self.context.dataset_manager.list_my_datasets.assert_called_once_with(
            verbose=True
        )

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
        self.context.config.root = "/test/root"
        with patch.object(fedbiomed.node.cli, "add_database") as m:
            args = self.parser.parse_args(["dataset", "add", "--mnist"])
            self.dataset_arg_pars.add(args)
            expected_path = os.path.join("/test/root", NODE_DATA_FOLDER)
            m.assert_called_once_with(
                self.context.dataset_manager, interactive=False, path=expected_path
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
        json_path = "/fake/dataset.json"
        with patch("builtins.open", mock_open(read_data=json.dumps(dataset_json))):
            with patch.object(fedbiomed.node.cli, "add_database") as m:
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                m.assert_called_once_with(
                    self.context.dataset_manager,
                    interactive=False,
                    path="/absolute/data/path",
                    data_type="csv",
                    description="Test dataset",
                    tags="#test",
                    name="TestData",
                    dataset_parameters=None,
                )

    def test_08_add_dataset_from_file_relative_path(self):
        """Tests _add_dataset_from_file prepends config.root for relative paths."""
        self.dataset_arg_pars.initialize()
        self.context.config.root = "/test/root"
        dataset_json = {
            "path": "relative/data",
            "data_type": "csv",
            "description": "Test",
            "tags": "#test",
            "name": "RelData",
        }
        json_path = "/fake/dataset.json"
        with patch("builtins.open", mock_open(read_data=json.dumps(dataset_json))):
            with patch.object(fedbiomed.node.cli, "add_database") as m:
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                call_kwargs = m.call_args[1]
                self.assertIn("/test/root", call_kwargs["path"])
                self.assertIn("relative", call_kwargs["path"])

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
        json_path = "/fake/dataset.json"
        with patch("builtins.open", mock_open(read_data=json.dumps(dataset_json))):
            with patch.dict(os.environ, {"FBM_TEST_DATA": "/env/path"}):
                with patch.object(fedbiomed.node.cli, "add_database") as m:
                    self.dataset_arg_pars._add_dataset_from_file(path=json_path)
                    call_kwargs = m.call_args[1]
                    self.assertIn("/env/path", call_kwargs["path"])

    def test_10_add_dataset_from_file_invalid_json(self):
        """Tests _add_dataset_from_file exits when the file is not valid JSON."""
        self.dataset_arg_pars.initialize()
        json_path = "/fake/dataset.json"
        with patch("builtins.open", mock_open(read_data="not valid json {{{")):
            with self.assertRaises(SystemExit):
                self.dataset_arg_pars._add_dataset_from_file(path=json_path)


class TestNodeControl(unittest.TestCase):
    """Test case for node control parser"""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = NodeControl(self.subparsers)
        self.context = MagicMock()
        self.control._context = self.context

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

    @patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    @patch("fedbiomed.node.node_pm.Node", autospec=True)
    def test_03_node_control__start(self, mock_node, mock_node_config):
        """Tests node start"""
        mock_node_config.return_value = MagicMock()
        mock_node.return_value.config.get.return_value = "test-node"
        mock_node.return_value.config.getbool.return_value = False
        mock_node.return_value.tp_security_manager = MagicMock()

        args = {"gpu": False}
        _start_node_process("config.ini", args)
        mock_node.return_value.task_manager.assert_called_once()

        with patch("fedbiomed.node.node_pm.logger") as logger:
            mock_node.return_value.task_manager.side_effect = FedbiomedError
            _start_node_process("config.ini", args)
            logger.critical.assert_called_once()
            logger.critical.reset_mock()

            mock_node.return_value.task_manager.side_effect = Exception
            _start_node_process("config.ini", args)
            logger.critical.assert_called_once()

    @patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    @patch("fedbiomed.node.node_pm.Node", autospec=True)
    def test_04_node_control__start_reports_failure_to_build_node(
        self, mock_node, mock_node_config
    ):
        """Tests that a node that cannot be built is reported, not raised"""
        mock_node_config.return_value = MagicMock()
        mock_node.side_effect = FedbiomedError(
            "FB619: no researcher certificate is registered"
        )

        with patch("fedbiomed.node.node_pm.logger") as logger:
            _start_node_process("config.ini", {"gpu": False})

            logger.critical.assert_called_once()
            self.assertIn("FB619", logger.critical.call_args[0][0])

    @patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    @patch("fedbiomed.node.node_pm.Node", autospec=True)
    def test_05_node_control__start_unexpected_failure_to_build_node(
        self, mock_node, mock_node_config
    ):
        """Tests reporting an unexpected error when there is no node to report with"""
        mock_node_config.return_value = MagicMock()
        mock_node.side_effect = Exception("unexpected")

        with patch("fedbiomed.node.node_pm.logger") as logger:
            _start_node_process("config.ini", {"gpu": False})

            logger.critical.assert_called_once()


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


class TestStartNodeProcess(unittest.TestCase):
    """Tests for the _start_node_process function."""

    @patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    @patch("fedbiomed.node.node_pm.Node")
    def test_01_start_node_training_plan_approval_with_default_plans(
        self, mock_node, mock_node_config
    ):
        """Tests tp_security_manager methods are called when approval + default plans are enabled."""
        mock_node_config.return_value = MagicMock()
        mock_node.return_value.config.getbool.return_value = True

        _start_node_process("config.ini", {"gpu": False})

        mock_node.return_value.tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
        mock_node.return_value.tp_security_manager.register_update_default_training_plans.assert_called_once()

    @patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    @patch("fedbiomed.node.node_pm.Node")
    def test_02_start_node_training_plan_approval_no_default_plans(
        self, mock_node, mock_node_config
    ):
        """Tests register_update_default_training_plans is NOT called when allow_default_training_plans is False."""
        mock_node_config.return_value = MagicMock()

        def _getbool(section, key):
            return key == "training_plan_approval"

        mock_node.return_value.config.getbool.side_effect = _getbool

        _start_node_process("config.ini", {"gpu": False})

        mock_node.return_value.tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
        mock_node.return_value.tp_security_manager.register_update_default_training_plans.assert_not_called()


class TestNodeCLI(unittest.TestCase):
    """Tests main NodeCLI"""

    @patch("fedbiomed.node.node.TrainingPlanSecurityManager")
    @patch("fedbiomed.node.node.DatasetManager")
    @patch("fedbiomed.node.cli.node_component.initiate")
    @patch("builtins.input")
    def test_01_node_cli_init(
        self, input_patch, mock_initiate, mock_dataset_manager, mock_tp_manager
    ):
        """Tests initialization"""
        input_patch.return_value = "y"
        mock_config = MagicMock()
        mock_config.get.return_value = "test-node-id"
        mock_initiate.return_value = mock_config
        mock_dataset_manager.return_value.list_my_datasets.return_value = []

        self.node_cli = NodeCLI()
        self.node_cli.parse_args(["--path", "/fake/fbm-node", "dataset", "list"])

        mock_initiate.assert_called_once_with(root=os.path.abspath("/fake/fbm-node"))
        mock_dataset_manager.assert_called_once()
        mock_dataset_manager.return_value.list_my_datasets.assert_called_once_with(
            verbose=True
        )


class TestNodeContext(unittest.TestCase):
    """Tests the local state the node CLI acts on"""

    @patch("fedbiomed.node.node.TrainingPlanSecurityManager")
    @patch("fedbiomed.node.node.DatasetManager")
    def setUp(self, mock_dataset, mock_tp):
        self.config = MagicMock()
        self.mock_dataset = mock_dataset
        self.mock_tp = mock_tp
        self.context = NodeContext(self.config)

    def test_01_exposes_config(self):
        """Tests that the context exposes the component configuration"""
        self.assertIs(self.context.config, self.config)

    def test_02_builds_both_managers_at_construction(self):
        """Tests that both managers are built when the context is created"""
        self.mock_dataset.assert_called_once()
        self.mock_tp.assert_called_once()
        self.assertIs(self.context.dataset_manager, self.mock_dataset.return_value)
        self.assertIs(self.context.tp_security_manager, self.mock_tp.return_value)


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


# ============================================================================
# END --- PYTEST TESTS FOR MEDICAL FOLDER DATASET CLI UTILITIES
# ============================================================================


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


@pytest.mark.parametrize(
    "argv, expected_node_args",
    [
        (
            ["start"],
            {
                "gpu": False,
                "gpu_num": 1,
                "gpu_only": False,
                "debug": False,
            },
        ),
        (
            ["start", "--force"],
            {
                "gpu": False,
                "gpu_num": 1,
                "gpu_only": False,
                "debug": False,
            },
        ),
        (
            ["start", "--gpu", "--gpu-num", "2", "--debug"],
            {
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": False,
                "debug": True,
            },
        ),
        (
            ["start", "--gpu-only"],
            {
                "gpu": True,
                "gpu_num": 1,
                "gpu_only": True,
                "debug": False,
            },
        ),
        (
            [
                "start",
                "--gpu",
                "--gpu-num",
                "2",
                "--gpu-only",
                "--debug",
                "--background",
            ],
            {
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": True,
                "debug": True,
            },
        ),
    ],
)
def test_node_control_start_builds_node_args_and_waits(
    mocker,
    argv,
    expected_node_args,
):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    control = NodeControl(subparsers)
    control.initialize()

    context = MagicMock()
    control._context = context

    mock_intro = mocker.patch("fedbiomed.node.cli.intro")

    mock_node_process_manager_cls = mocker.patch(
        "fedbiomed.node.cli.NodeProcessManager"
    )
    mock_node_process_manager = mock_node_process_manager_cls.return_value

    args = parser.parse_args(argv)

    control.start(args)

    mock_intro.assert_called_once_with()
    mock_node_process_manager_cls.assert_called_once_with(context.config)

    mock_node_process_manager.start.assert_called_once_with(
        node_args=expected_node_args,
        background=args.background,
        actor={"source": "cli"},
        force=args.force,
    )

    mock_node_process_manager.stop.assert_not_called()


def test_node_control_start_keyboard_interrupt_stops_node(mocker):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    control = NodeControl(subparsers)
    control.initialize()

    context = MagicMock()
    control._context = context

    mock_intro = mocker.patch("fedbiomed.node.cli.intro")

    mock_node_process_manager_cls = mocker.patch(
        "fedbiomed.node.cli.NodeProcessManager"
    )
    mock_node_process_manager = mock_node_process_manager_cls.return_value
    mock_node_process_manager.start.side_effect = KeyboardInterrupt

    args = parser.parse_args(["start"])

    with pytest.raises(SystemExit) as exc:
        control.start(args)

    assert exc.value.code == 0

    mock_intro.assert_called_once_with()
    mock_node_process_manager_cls.assert_called_once_with(context.config)

    mock_node_process_manager.start.assert_called_once_with(
        node_args={
            "gpu": False,
            "gpu_num": 1,
            "gpu_only": False,
            "debug": False,
        },
        background=False,
        actor={"source": "cli"},
        force=False,
    )

    mock_node_process_manager.stop.assert_called_once_with(
        actor={"source": "cli"},
        reason="keyboard_interrupt",
    )


@pytest.mark.parametrize(
    "argv, expected_node_args, expected_background",
    [
        (["restart"], {}, None),
        (
            ["restart", "--gpu", "--gpu-num", "2", "--debug", "--background"],
            {"gpu": True, "gpu_num": 2, "debug": True},
            True,
        ),
        (
            ["restart", "--gpu-only"],
            {"gpu": True, "gpu_only": True},
            None,
        ),
    ],
)
def test_node_control_restart_passes_only_supplied_overrides(
    mocker,
    argv,
    expected_node_args,
    expected_background,
):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    control = NodeControl(subparsers)
    control.initialize()

    context = MagicMock()
    control._context = context

    mock_node_process_manager_cls = mocker.patch(
        "fedbiomed.node.cli.NodeProcessManager"
    )
    mock_node_process_manager = mock_node_process_manager_cls.return_value

    control.restart(parser.parse_args(argv))

    mock_node_process_manager.restart.assert_called_once_with(
        node_args=expected_node_args,
        background=expected_background,
        actor={"source": "cli"},
        reason="cli_restart_command",
    )


if __name__ == "__main__":
    unittest.main()
