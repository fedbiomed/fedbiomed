import argparse
import configparser
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fedbiomed
import fedbiomed.node.cli_utils
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import (
    DatasetArgumentParser,
    NodeCLI,
    NodeControl,
    TrainingPlanArgumentParser,
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

        self.node.dataset_manager.list_my_datasets.return_value = []
        self.node.dataset_manager.list_my_datasets.assert_called_once_with(verbose=True)


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

    @patch("fedbiomed.node.cli.Process")
    def test_02_node_control_start(self, process):
        self.control.initialize()
        args = self.parser.parse_args(["start"])
        os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"

        self.control.start(args)
        process.assert_called_once()

        process.return_value.join.side_effect = [KeyboardInterrupt, None]
        process.return_value.is_alive.side_effect = [True, False, True, True, False]
        with self.assertRaises(SystemExit):
            self.control.start(args)

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
        interactive=True, dataset_parameters=None, dlp=None
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
def test_add_medical_folder_dataset_with_demographics_interactive(
    mock_input,
    mock_controller_class,
    mock_validated_path,
    temp_medical_data,
    mock_medical_folder_controller,
):
    """Test adding medical folder dataset with demographics file in interactive mode."""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.side_effect = [test_path, csv_path]
    mock_input.side_effect = ["y", "1"]  # Yes to demographics, index column 1
    mock_controller_class.return_value = mock_medical_folder_controller

    # Call function
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        interactive=True, dataset_parameters={"existing_param": "value"}, dlp=None
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
        interactive=True, dataset_parameters=None, dlp=None
    )

    # Assertions
    assert result_params["index_col"] == 2
    mock_warn.assert_called_once_with(
        "Please input a numeric value (integer)", stacklevel=1
    )


@patch("fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input")
@patch("fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController")
@patch("builtins.input")
def test_add_medical_folder_dataset_non_interactive_with_demographics(
    mock_input, mock_controller_class, mock_validated_path, temp_medical_data
):
    """Test non-interactive mode with demographics file."""
    temp_dir, test_path, csv_path = temp_medical_data

    # Setup mocks
    mock_validated_path.side_effect = [test_path, csv_path]
    mock_input.side_effect = ["y", "0"]  # Yes to demographics, index 0

    mock_controller = MagicMock()
    mock_controller.df_dir = {"modality": MagicMock()}
    mock_controller.df_dir["modality"].unique.return_value = ["flair", "t1ce"]
    mock_controller.demographics_column_names.return_value = ["patient_id", "diagnosis"]
    mock_controller_class.return_value = mock_controller

    # Call function in non-interactive mode
    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        interactive=False, dataset_parameters=None, dlp=None
    )

    # Assertions
    assert result_params["index_col"] == 0
    assert result_params["tabular_file"] == csv_path


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
        interactive=True, dataset_parameters=None, dlp=existing_dlp
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
        interactive=True, dataset_parameters=existing_params, dlp=None
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


if __name__ == "__main__":
    unittest.main()
