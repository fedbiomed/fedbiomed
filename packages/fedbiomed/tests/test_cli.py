import argparse
import json
import os
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import fedbiomed
import fedbiomed.node.cli_utils
import fedbiomed.node.cli_utils._medical_folder_dataset as medical_folder_cli
import pytest
from fedbiomed.common.cli import CommonCLI
from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import (
    DatasetArgumentParser,
    NodeCLI,
    NodeControl,
    TrainingPlanArgumentParser,
    intro,
)
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
    get_map_modalities2folders_from_cli,
)
from fedbiomed.node.node import (
    CertificateDiagnostic,
    DiagnosticSeverity,
    NodeContext,
)
from fedbiomed.node.node_pm import (
    NodeConnectionStateManager,
    NodeState,
    _node_signal_trigger_term,
    _start_node_process,
)

# ============================================================================
# SHARED FIXTURES AND HELPERS
# ============================================================================


@pytest.fixture
def temp_medical_data():
    """Fixture to provide fake test data paths."""
    return "/fake/tmp", "/fake/medical_data", "/fake/demographics.csv"


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


# ============================================================================
# TRAINING PLAN ARGUMENT PARSER
# ============================================================================


@pytest.fixture
def tp_cli():
    """Initialized training plan parser, its argument parser and subparsers."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    tp_arg_pars = TrainingPlanArgumentParser(subparsers)
    tp_arg_pars._context = MagicMock()
    tp_arg_pars.initialize()
    return SimpleNamespace(parser=parser, subparsers=subparsers, tp=tp_arg_pars)


def test_training_plan_argument_parser_initialize(tp_cli):
    """Tests training plan parser initialization"""
    assert "training-plan" in tp_cli.subparsers.choices

    tp_choices = (
        tp_cli.subparsers.choices["training-plan"]._subparsers._group_actions[0].choices
    )
    for command in ("approve", "list", "delete", "reject", "view", "update"):
        assert command in tp_choices


def test_training_plan_argument_parser_execute(tp_cli, mocker):
    """Tests training plan argument parser actions"""
    args = tp_cli.parser.parse_args(["training-plan", "delete"])
    tp_cli.tp.delete(args)
    tp_cli.tp.list()
    tp_cli.tp.update()

    view = mocker.patch.object(fedbiomed.node.cli, "view_training_plan")
    tp_cli.tp.view()
    view.assert_called_once()

    register = mocker.patch.object(fedbiomed.node.cli, "register_training_plan")
    tp_cli.tp.register()
    register.assert_called_once()

    reject = mocker.patch.object(fedbiomed.node.cli, "reject_training_plan")
    tp_cli.tp.reject(tp_cli.parser.parse_args(["training-plan", "reject"]))
    reject.assert_called_once()


def test_training_plan_argument_parser_approve(tp_cli, mocker):
    """Tests training plan approve action calls approve_training_plan."""
    approve = mocker.patch.object(fedbiomed.node.cli, "approve_training_plan")
    tp_cli.tp.approve(tp_cli.parser.parse_args(["training-plan", "approve"]))
    approve.assert_called_once()


# ============================================================================
# DATASET ARGUMENT PARSER
# ============================================================================


@pytest.fixture
def dataset_cli():
    """Initialized dataset parser, its argument parser, subparsers and context."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    dataset_arg_pars = DatasetArgumentParser(subparsers)
    context = MagicMock()
    dataset_arg_pars._context = context
    dataset_arg_pars.initialize()
    return SimpleNamespace(
        parser=parser, subparsers=subparsers, dataset=dataset_arg_pars, context=context
    )


def _dataset_file(tmp_path, **fields):
    """Writes a dataset description file and returns its path."""
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(fields))
    return str(path)


def test_dataset_argument_parser_initialize(dataset_cli):
    """Test initialization"""
    assert "dataset" in dataset_cli.subparsers.choices
    dataset_choices = (
        dataset_cli.subparsers.choices["dataset"]._subparsers._group_actions[0].choices
    )

    for command in ("add", "list", "delete"):
        assert command in dataset_choices

    assert "--mnist" in dataset_choices["add"]._option_string_actions
    assert "--file" in dataset_choices["add"]._option_string_actions

    assert "--all" in dataset_choices["delete"]._option_string_actions
    assert "--mnist" in dataset_choices["delete"]._option_string_actions


def test_dataset_argument_parser_add(dataset_cli, mocker):
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")

    dataset_cli.dataset.add(dataset_cli.parser.parse_args(["dataset", "add"]))
    add_database.assert_called_once()
    add_database.reset_mock()

    args = dataset_cli.parser.parse_args(["dataset", "add", "--mnist", "test"])
    dataset_cli.dataset.add(args)
    add_database.assert_called_once_with(
        dataset_cli.context.dataset_manager, interactive=False, path="test"
    )


def test_dataset_argument_parser_delete(dataset_cli, mocker):
    delete_database = mocker.patch.object(fedbiomed.node.cli, "delete_database")
    delete_all_database = mocker.patch.object(fedbiomed.node.cli, "delete_all_database")
    manager = dataset_cli.context.dataset_manager

    dataset_cli.dataset.delete(dataset_cli.parser.parse_args(["dataset", "delete"]))
    delete_database.assert_called_once_with(manager)
    delete_database.reset_mock()

    args = dataset_cli.parser.parse_args(["dataset", "delete", "--all"])
    dataset_cli.dataset.delete(args)
    delete_all_database.assert_called_once_with(manager)

    args = dataset_cli.parser.parse_args(["dataset", "delete", "--mnist"])
    dataset_cli.dataset.delete(args)
    delete_database.assert_called_once_with(manager, interactive=False)


def test_dataset_argument_parser_list(dataset_cli):
    dataset_cli.dataset.list(dataset_cli.parser.parse_args(["dataset", "list"]))

    dataset_cli.context.dataset_manager.list_my_datasets.assert_called_once_with(
        verbose=True
    )


def test_dataset_argument_parser_add_file(dataset_cli, mocker):
    """Tests add() dispatches to _add_dataset_from_file when --file is given."""
    add_from_file = mocker.patch.object(dataset_cli.dataset, "_add_dataset_from_file")
    args = dataset_cli.parser.parse_args(
        ["dataset", "add", "--file", "/path/to/dataset.json"]
    )

    dataset_cli.dataset.add(args)

    add_from_file.assert_called_once_with(path="/path/to/dataset.json")


def test_dataset_argument_parser_add_mnist_default_path(dataset_cli, mocker):
    """Tests add() uses node data folder when --mnist is given without a value."""
    dataset_cli.context.config.root = "/test/root"
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")

    dataset_cli.dataset.add(
        dataset_cli.parser.parse_args(["dataset", "add", "--mnist"])
    )

    add_database.assert_called_once_with(
        dataset_cli.context.dataset_manager,
        interactive=False,
        path=os.path.join("/test/root", NODE_DATA_FOLDER),
    )


def test_add_dataset_from_file_absolute_path(dataset_cli, mocker, tmp_path):
    """Tests _add_dataset_from_file preserves absolute paths."""
    json_path = _dataset_file(
        tmp_path,
        path="/absolute/data/path",
        data_type="csv",
        description="Test dataset",
        tags="#test",
        name="TestData",
    )
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")

    dataset_cli.dataset._add_dataset_from_file(path=json_path)

    add_database.assert_called_once_with(
        dataset_cli.context.dataset_manager,
        interactive=False,
        path="/absolute/data/path",
        data_type="csv",
        description="Test dataset",
        tags="#test",
        name="TestData",
        dataset_parameters=None,
    )


def test_add_dataset_from_file_relative_path(dataset_cli, mocker, tmp_path):
    """Tests _add_dataset_from_file prepends config.root for relative paths."""
    dataset_cli.context.config.root = "/test/root"
    json_path = _dataset_file(
        tmp_path,
        path="relative/data",
        data_type="csv",
        description="Test",
        tags="#test",
        name="RelData",
    )
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")

    dataset_cli.dataset._add_dataset_from_file(path=json_path)

    path = add_database.call_args.kwargs["path"]
    assert "/test/root" in path
    assert "relative" in path


def test_add_dataset_from_file_env_var_path(dataset_cli, mocker, monkeypatch, tmp_path):
    """Tests _add_dataset_from_file expands environment variable in path."""
    json_path = _dataset_file(
        tmp_path,
        path="$FBM_TEST_DATA/data",
        data_type="csv",
        description="Test",
        tags="#test",
        name="EnvData",
    )
    monkeypatch.setenv("FBM_TEST_DATA", "/env/path")
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")

    dataset_cli.dataset._add_dataset_from_file(path=json_path)

    assert "/env/path" in add_database.call_args.kwargs["path"]


def test_add_dataset_from_file_invalid_json(dataset_cli, tmp_path):
    """Tests _add_dataset_from_file exits when the file is not valid JSON."""
    json_path = tmp_path / "dataset.json"
    json_path.write_text("not valid json {{{")

    with pytest.raises(SystemExit):
        dataset_cli.dataset._add_dataset_from_file(path=str(json_path))


# ============================================================================
# NODE CONTROL AND NODE PROCESS START
# ============================================================================


def _node_config_in(tmp_dir, mock_node_config):
    """Points the patched node configuration at a database in `tmp_dir`.

    Starting a node records its state, so the configuration has to name a real file.
    """
    config = mock_node_config.return_value
    config.get.return_value = "test-node"
    config.getbool.return_value = False
    config.getpath.return_value = os.path.join(tmp_dir, "node_db.json")

    return config


@pytest.fixture
def node_config(mocker, tmp_path):
    """Patched `NodeConfig` of the process start, pointing at a database in `tmp_path`."""
    mock_node_config = mocker.patch("fedbiomed.node.node_pm.NodeConfig", autospec=True)
    return _node_config_in(str(tmp_path), mock_node_config)


def test_node_control_initialize():
    """Tests initialize"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    control = NodeControl(subparsers)
    control._context = MagicMock()

    control.initialize()

    assert "start" in subparsers.choices
    for option in ("--gpu-only", "--gpu-num", "--gpu"):
        assert option in subparsers.choices["start"]._option_string_actions


def test_node_control__start(node_config, mocker):
    """Tests node start"""
    mock_node = mocker.patch("fedbiomed.node.node_pm.Node", autospec=True)
    mock_node.return_value.config.get.return_value = "test-node"
    mock_node.return_value.config.getbool.return_value = False
    mock_node.return_value.tp_security_manager = MagicMock()

    args = {"gpu": False}
    _start_node_process("config.ini", args)
    mock_node.return_value.task_manager.assert_called_once()
    # A run in progress is recorded before the node is built, so the state of
    # the previous one is not read as the current one
    state = NodeConnectionStateManager(node_config).get_connection_state()
    assert state["operation"] == "node_starting"

    logger = mocker.patch("fedbiomed.node.node_pm.logger")
    mock_node.return_value.task_manager.side_effect = FedbiomedError
    _start_node_process("config.ini", args)
    logger.critical.assert_called_once()
    logger.critical.reset_mock()

    mock_node.return_value.task_manager.side_effect = Exception
    _start_node_process("config.ini", args)
    logger.critical.assert_called_once()


def test_node_control__start_reports_failure_to_build_node(node_config, mocker):
    """Tests that a node that cannot be built is reported, not raised"""
    mock_node = mocker.patch("fedbiomed.node.node_pm.Node", autospec=True)
    mock_node.side_effect = FedbiomedError(
        "FB619: no researcher certificate is registered"
    )
    logger = mocker.patch("fedbiomed.node.node_pm.logger")

    _start_node_process("config.ini", {"gpu": False})

    logger.critical.assert_called_once()
    assert "FB619" in logger.critical.call_args[0][0]


def test_node_control__start_unexpected_failure_to_build_node(node_config, mocker):
    """Tests reporting an unexpected error when there is no node to report with"""
    mock_node = mocker.patch("fedbiomed.node.node_pm.Node", autospec=True)
    mock_node.side_effect = Exception("unexpected")
    logger = mocker.patch("fedbiomed.node.node_pm.logger")

    _start_node_process("config.ini", {"gpu": False})

    logger.critical.assert_called_once()


def test_start_node_training_plan_approval_with_default_plans(node_config, mocker):
    """Tests tp_security_manager methods are called when approval + default plans are enabled."""
    mock_node = mocker.patch("fedbiomed.node.node_pm.Node")
    mock_node.return_value.config.getbool.return_value = True

    _start_node_process("config.ini", {"gpu": False})

    tp_security_manager = mock_node.return_value.tp_security_manager
    tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
    tp_security_manager.register_update_default_training_plans.assert_called_once()


def test_start_node_training_plan_approval_no_default_plans(node_config, mocker):
    """Tests register_update_default_training_plans is NOT called when allow_default_training_plans is False."""
    mock_node = mocker.patch("fedbiomed.node.node_pm.Node")
    mock_node.return_value.config.getbool.side_effect = (
        lambda section, key: key == "training_plan_approval"
    )

    _start_node_process("config.ini", {"gpu": False})

    tp_security_manager = mock_node.return_value.tp_security_manager
    tp_security_manager.check_hashes_for_registered_training_plans.assert_called_once()
    tp_security_manager.register_update_default_training_plans.assert_not_called()


# ============================================================================
# NODE CLI AND CONTEXT
# ============================================================================


def test_node_cli_init(mocker):
    """Tests initialization"""
    mocker.patch("builtins.input", return_value="y")
    mock_initiate = mocker.patch("fedbiomed.node.cli.node_component.initiate")
    mock_dataset_manager = mocker.patch("fedbiomed.node.node.DatasetManager")
    mocker.patch("fedbiomed.node.node.TrainingPlanSecurityManager")
    mock_config = MagicMock()
    mock_config.get.return_value = "test-node-id"
    mock_initiate.return_value = mock_config
    mock_dataset_manager.return_value.list_my_datasets.return_value = []

    NodeCLI().parse_args(["--path", "/fake/fbm-node", "dataset", "list"])

    mock_initiate.assert_called_once_with(root=os.path.abspath("/fake/fbm-node"))
    mock_dataset_manager.assert_called_once()
    mock_dataset_manager.return_value.list_my_datasets.assert_called_once_with(
        verbose=True
    )


@pytest.fixture
def node_context(mocker):
    """A `NodeContext` built with its two managers patched."""
    mock_dataset = mocker.patch("fedbiomed.node.node.DatasetManager")
    mock_tp = mocker.patch("fedbiomed.node.node.TrainingPlanSecurityManager")
    config = MagicMock()
    return SimpleNamespace(
        context=NodeContext(config), config=config, dataset=mock_dataset, tp=mock_tp
    )


def test_node_context_exposes_config(node_context):
    """Tests that the context exposes the component configuration"""
    assert node_context.context.config is node_context.config


def test_node_context_builds_both_managers_at_construction(node_context):
    """Tests that both managers are built when the context is created"""
    node_context.dataset.assert_called_once()
    node_context.tp.assert_called_once()
    assert node_context.context.dataset_manager is node_context.dataset.return_value
    assert node_context.context.tp_security_manager is node_context.tp.return_value


# ============================================================================
# MEDICAL FOLDER DATASET CLI UTILITIES
# ============================================================================


@pytest.fixture
def medical_folder_mocks(mocker):
    """Patched path prompt, controller class and input of the medical folder CLI."""
    return SimpleNamespace(
        validated_path=mocker.patch(
            "fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input"
        ),
        controller_class=mocker.patch(
            "fedbiomed.node.cli_utils._medical_folder_dataset.MedicalFolderController"
        ),
        input=mocker.patch("builtins.input"),
    )


def test_add_medical_folder_dataset_no_demographics(
    medical_folder_mocks, temp_medical_data, mock_medical_folder_controller
):
    """Test adding medical folder dataset without demographics file."""
    temp_dir, test_path, csv_path = temp_medical_data
    mocks = medical_folder_mocks
    mocks.validated_path.return_value = test_path
    mocks.input.return_value = "n"  # No demographics file
    mocks.controller_class.return_value = mock_medical_folder_controller

    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=None, dlp=None
    )

    assert result_path == test_path
    assert result_params == {}
    assert result_dlp is None

    mocks.validated_path.assert_called_once_with(type="dir", initialdir=None)
    mocks.controller_class.assert_called_once_with(test_path)
    mocks.input.assert_called_once_with(
        "\nWould you like to select a demographics csv file? [y/N]\n"
    )


def test_add_medical_folder_dataset_demographics(
    medical_folder_mocks, temp_medical_data, mock_medical_folder_controller
):
    """Test adding medical folder dataset with demographics file"""
    temp_dir, test_path, csv_path = temp_medical_data
    mocks = medical_folder_mocks
    mocks.validated_path.side_effect = [test_path, csv_path]
    mocks.input.side_effect = ["y", "1"]  # Yes to demographics, index column 1
    mocks.controller_class.return_value = mock_medical_folder_controller

    result_path, result_params, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters={"existing_param": "value"}, dlp=None
    )

    assert result_path == test_path
    assert result_params == {
        "existing_param": "value",
        "tabular_file": csv_path,
        "index_col": 1,
    }
    assert result_dlp is None

    assert mocks.validated_path.call_count == 2
    mock_medical_folder_controller.demographics_column_names.assert_called_once_with(
        csv_path
    )


def test_add_medical_folder_dataset_invalid_index_then_valid(
    medical_folder_mocks, temp_medical_data, mock_medical_folder_controller
):
    """Test handling invalid index input followed by valid input."""
    temp_dir, test_path, csv_path = temp_medical_data
    mocks = medical_folder_mocks
    mocks.validated_path.side_effect = [test_path, csv_path]
    # Yes, invalid input, then valid index
    mocks.input.side_effect = ["y", "invalid", "2"]
    mocks.controller_class.return_value = mock_medical_folder_controller

    with pytest.warns(UserWarning, match="Please input a numeric value") as warned:
        _, result_params, _ = add_medical_folder_dataset_from_cli(
            dataset_parameters=None, dlp=None
        )

    assert result_params["index_col"] == 2
    assert len(warned) == 1


def test_add_medical_folder_dataset_with_existing_dlp(
    medical_folder_mocks, temp_medical_data, mock_medical_folder_controller
):
    """Test function with existing DataLoadingPlan."""
    from fedbiomed.common.dataloadingplan import DataLoadingPlan

    temp_dir, test_path, csv_path = temp_medical_data
    mocks = medical_folder_mocks
    mocks.validated_path.return_value = test_path
    mocks.input.return_value = "n"  # No demographics
    mocks.controller_class.return_value = mock_medical_folder_controller
    existing_dlp = DataLoadingPlan()

    _, _, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=None, dlp=existing_dlp
    )

    assert result_dlp == existing_dlp


def test_add_medical_folder_dataset_preserves_existing_parameters(
    medical_folder_mocks, temp_medical_data, mock_medical_folder_controller
):
    """Test that existing dataset parameters are preserved."""
    temp_dir, test_path, csv_path = temp_medical_data
    mocks = medical_folder_mocks
    mocks.validated_path.return_value = test_path
    mocks.input.return_value = "n"  # No demographics
    mocks.controller_class.return_value = mock_medical_folder_controller
    existing_params = {"custom_param": "value", "another_param": 123}

    _, result_params, _ = add_medical_folder_dataset_from_cli(
        dataset_parameters=existing_params, dlp=None
    )

    assert result_params["custom_param"] == "value"
    assert result_params["another_param"] == 123
    assert len(result_params) == 2  # No additional params added


# Test scenarios for modality mapping with parametrization
@pytest.mark.parametrize(
    "modality_folder_names,inputs,expected_map,expect_warnings",
    [
        # basic mapping
        (["Should map to T1"], ["1"], {"T1": ["Should map to T1"]}, False),
        # wrong inputs then correct
        (
            ["Should map to T1"],
            ["wrong", "5", "1"],
            {"T1": ["Should map to T1"]},
            True,
        ),
        # add a new modality
        (
            ["Should map to T1", "Should map to Tnew"],
            ["1", "0", "Tnew", "y", "4"],
            {"T1": ["Should map to T1"], "Tnew": ["Should map to Tnew"]},
            False,
        ),
        # complex scenario
        (
            [
                "Should map to T1",
                "Should map to Tnew",
                "Should also map to T1",
                "Should map to T2",
            ],
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
def test_get_map_modalities2folders_from_cli_scenarios(
    monkeypatch, modality_folder_names, inputs, expected_map, expect_warnings
):
    """Test various scenarios for modality to folder mapping."""
    answers = iter(inputs)
    monkeypatch.setattr(
        medical_folder_cli, "input", lambda _: next(answers), raising=False
    )

    if expect_warnings:
        with pytest.warns(UserWarning):
            dlb = get_map_modalities2folders_from_cli(modality_folder_names)
    else:
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)

    assert dlb.map == expected_map


# ============================================================================
# END --- PYTEST TESTS FOR MEDICAL FOLDER DATASET CLI UTILITIES
# ============================================================================


def test_intro(monkeypatch, capsys):
    """Tests intro() prints the active node ID from the environment."""
    monkeypatch.setenv("FEDBIOMED_ACTIVE_NODE_ID", "test-node-id")

    intro()

    assert "test-node-id" in capsys.readouterr().out


def test_node_signal_trigger_term(mocker):
    """Tests _node_signal_trigger_term sends SIGTERM to the current process."""
    mock_kill = mocker.patch("os.kill")

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


def _checking_cli(mocker, diagnostics):
    """A `NodeCLI` whose certificate diagnostics are the ones given."""
    mocker.patch("fedbiomed.node.cli.node_component.initiate")
    cli = NodeCLI()
    cli.config = MagicMock()
    cli.config.getbool.return_value = True
    cli.config.get.side_effect = lambda section, key: {
        ("researcher", "ip"): "10.0.0.9",
        ("researcher", "port"): "50051",
    }[(section, key)]
    mocker.patch("fedbiomed.node.cli.certificate_diagnostics", return_value=diagnostics)
    return cli


def test_node_cli_certificate_check_is_node_only(mocker):
    """A researcher's readiness is a different question, so it has no `check`."""
    mocker.patch("fedbiomed.node.cli.node_component.initiate")
    node_choices = NodeCLI()._certificate_sub_parsers.choices

    assert "check" in node_choices
    assert node_choices["check"]._defaults["func"].__func__.__name__ == (
        "_check_certificate"
    )

    common = CommonCLI()
    common.initialize_certificate_parser()
    assert "check" not in common._certificate_sub_parsers.choices
    # The commands both components share are still there.
    assert "register" in common._certificate_sub_parsers.choices


def test_node_cli_certificate_check_exits_on_a_problem(mocker, capsys):
    """A non-zero exit says the node would not start, for a deployment script."""
    cli = _checking_cli(
        mocker,
        [
            CertificateDiagnostic(DiagnosticSeverity.PROBLEM, "nothing registered"),
            CertificateDiagnostic(DiagnosticSeverity.WARNING, "expires soon"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli._check_certificate(argparse.Namespace())

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "1 problem(s) stop this node from starting" in out
    assert "nothing registered" in out
    assert "expires soon" in out


def test_node_cli_certificate_check_succeeds_on_warnings(mocker, capsys):
    """Warnings are reported but do not fail the command."""
    cli = _checking_cli(
        mocker, [CertificateDiagnostic(DiagnosticSeverity.WARNING, "host not in SAN")]
    )

    cli._check_certificate(argparse.Namespace())

    out = capsys.readouterr().out
    assert "1 warning(s)" in out
    assert "host not in SAN" in out
    assert "problem(s)" not in out


def test_node_cli_certificate_check_reports_a_clean_node(mocker, capsys):
    """Nothing to report is itself the answer."""
    cli = _checking_cli(mocker, [])

    cli._check_certificate(argparse.Namespace())

    assert "No certificate problem found" in capsys.readouterr().out


def _status_control(mocker, connection_state, history=()):
    """A `NodeControl` whose process and connection state are the ones given."""
    parser = argparse.ArgumentParser()
    control = NodeControl(parser.add_subparsers())
    control.initialize()
    control._context = MagicMock()

    mocker.patch(
        "fedbiomed.node.cli.NodeProcessManager"
    ).return_value.get_status.return_value = NodeState.STOPPED
    manager = mocker.patch("fedbiomed.node.cli.NodeConnectionStateManager")
    manager.return_value.get_connection_state.return_value = connection_state
    manager.return_value.get_connection_history.return_value = list(history)

    return control, parser


def _connection_entry(**overrides):
    fields = {
        "node_id": "node-1",
        "state": "connected",
        "host": "10.0.0.9",
        "port": "50051",
        "mtls": True,
        "updated_at": "2026-09-03T07:18:32Z",
    }
    return {**fields, **overrides}


def test_node_control_status_reports_the_recorded_connection(mocker, capsys):
    """The connection the node last observed is reported beside its process state."""
    control, parser = _status_control(mocker, _connection_entry())

    control.status(parser.parse_args(["status"]))

    out = capsys.readouterr().out
    assert "Node process" in out and "stopped" in out
    assert "10.0.0.9" in out and "50051" in out
    assert "Mutual authentication (mTLS)" in out and "On" in out
    assert "Identity verified" in out and "Unknown" in out
    for field in ("State", "Host", "Port", "Reason", "Connected since", "Last update"):
        assert field in out
    # Nothing is written while the node is stopped, so say the state is not current.
    assert "not running" in out


@pytest.mark.parametrize(
    "state, code", [(NodeState.RUNNING, "\033[32m"), (NodeState.STOPPED, "\033[31m")]
)
def test_node_control_status_colors_the_process_state_on_a_terminal(
    mocker, capsys, monkeypatch, state, code
):
    """Running is green and stopped is red, but only on a terminal."""
    control, parser = _status_control(mocker, None)
    fedbiomed_pm = mocker.patch("fedbiomed.node.cli.NodeProcessManager")
    fedbiomed_pm.return_value.get_status.return_value = state
    monkeypatch.delenv("NO_COLOR", raising=False)

    mocker.patch("fedbiomed.node.cli.sys.stdout.isatty", return_value=True)
    control.status(parser.parse_args(["status"]))
    assert f"{code}{state.value}\033[0m" in capsys.readouterr().out

    mocker.patch("fedbiomed.node.cli.sys.stdout.isatty", return_value=False)
    control.status(parser.parse_args(["status"]))
    assert "\033[" not in capsys.readouterr().out


def test_node_control_status_without_a_recorded_connection(mocker, capsys):
    """A node that never connected says so rather than reporting an empty state."""
    control, parser = _status_control(mocker, None)

    control.status(parser.parse_args(["status"]))

    assert "none recorded yet" in capsys.readouterr().out


def test_node_control_status_lists_history_only_when_asked(mocker, capsys):
    """The transitions are noise on a routine status check, so `--history` asks."""
    history = [
        _connection_entry(operation="handshake_complete"),
        _connection_entry(state="disconnected", operation="node_starting"),
    ]
    control, parser = _status_control(mocker, history[0], history)

    control.status(parser.parse_args(["status"]))
    assert "Recent connection state changes" not in capsys.readouterr().out

    control.status(parser.parse_args(["status", "--history"]))
    out = capsys.readouterr().out
    assert "Recent connection state changes" in out
    assert "handshake_complete" in out
    assert "node_starting" in out


def test_delete_database_removes_mnist_with_its_derived_datasets():
    """Deleting from the CLI also deletes the datasets derived from it."""
    manager = MagicMock()
    manager.list_my_datasets.return_value = [
        {"name": "Other", "dataset_id": "d1"},
        {"name": "MNIST", "dataset_id": "d2"},
    ]

    fedbiomed.node.cli_utils.delete_database(manager, interactive=False)

    manager.delete_dataset_by_id.assert_called_once_with("d2", recursive=True)


def test_delete_all_database_removes_derived_datasets():
    manager = MagicMock()
    manager.list_my_datasets.return_value = [{"dataset_id": "d1"}, {"dataset_id": "d2"}]

    fedbiomed.node.cli_utils.delete_all_database(manager)

    assert manager.delete_dataset_by_id.call_args_list == [
        call("d1", recursive=True),
        call("d2", recursive=True),
    ]
