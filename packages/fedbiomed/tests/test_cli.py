import json
import os
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import fedbiomed
import pytest
from fedbiomed.common.cli import CommonCLI
from fedbiomed.common.constants import NODE_DATA_FOLDER, TrainingPlanStatus
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.exceptions import (
    FedbiomedDatasetError,
    FedbiomedDatasetManagerError,
    FedbiomedError,
)
from fedbiomed.node.cli import NodeCLI, intro
from fedbiomed.node.cli_utils._database import (
    add_database,
    delete_all_database,
    delete_database,
)
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
    get_map_modalities2folders_from_cli,
)
from fedbiomed.node.cli_utils._training_plan_management import (
    delete_training_plan,
    update_training_plan,
)
from fedbiomed.node.dataset_manager._db_dataclasses import NodeConnectionStateEntry
from fedbiomed.node.node import CertificateDiagnostic, DiagnosticSeverity
from fedbiomed.node.node_pm import NodeState

NODE_ROOT = os.path.abspath("/fake/fbm-node")
NODE_DATA_PATH = os.path.join(NODE_ROOT, NODE_DATA_FOLDER)
# Node arguments `start` passes when no option is given
START_DEFAULTS = {"gpu": False, "gpu_num": 1, "gpu_only": False, "debug": False}
CLI_ACTOR = {"source": "cli"}
# Paths the mocked medical folder prompts return; nothing is read from them
MEDICAL_DATA_PATH = "/fake/medical_data"
DEMOGRAPHICS_PATH = "/fake/demographics.csv"
# Valid dataset values given to the CLI's `add_database`
CLI_DATASET = {
    "name": "data",
    "tags": "tag1",
    "description": "desc",
    "data_type": "csv",
}


# ============================================================================
# NODE CLI COMMANDS
# ============================================================================


@pytest.fixture
def node_cli(mocker, monkeypatch):
    """Runs node CLI commands on a node at `NODE_ROOT` whose managers are mocked."""
    initiate = mocker.patch("fedbiomed.node.cli.node_component.initiate")
    config = initiate.return_value
    config.root = NODE_ROOT
    config.get.return_value = "test-node-id"
    # Restored after the test: opening a node sets them
    monkeypatch.delenv("FBM_NODE_COMPONENT_ROOT", raising=False)
    monkeypatch.delenv("FEDBIOMED_ACTIVE_NODE_ID", raising=False)
    datasets = mocker.patch("fedbiomed.node.node.DatasetManager").return_value
    plans = mocker.patch("fedbiomed.node.node.TrainingPlanSecurityManager").return_value
    # Confirms using `NODE_ROOT`, which does not exist
    mocker.patch("builtins.input", return_value="y")

    def run(*argv):
        NodeCLI().parse_args(["--path", NODE_ROOT, *argv])

    return SimpleNamespace(
        run=run, initiate=initiate, config=config, datasets=datasets, plans=plans
    )


def test_node_cli_opens_the_node_at_the_given_path(node_cli):
    node_cli.run("dataset", "list")

    node_cli.initiate.assert_called_once_with(root=NODE_ROOT)


@pytest.mark.parametrize(
    "argv, action, expected",
    [
        (["delete", "--id", "tp-1"], "delete_training_plan", {"id": "tp-1"}),
        (["approve", "--id", "tp-1"], "approve_training_plan", {"id": "tp-1"}),
        (
            ["reject", "--id", "tp-1", "--notes", "unsafe"],
            "reject_training_plan",
            {"id": "tp-1", "notes": "unsafe"},
        ),
        (
            ["reject", "--id", "tp-1"],
            "reject_training_plan",
            {"id": "tp-1", "notes": "No notes provided."},
        ),
        (["register"], "register_training_plan", {}),
        (["view"], "view_training_plan", {}),
        (["update"], "update_training_plan", {}),
    ],
    ids=[
        "delete",
        "approve",
        "reject",
        "reject_default_notes",
        "register",
        "view",
        "update",
    ],
)
def test_training_plan_command_runs_its_action(
    node_cli, mocker, argv, action, expected
):
    run_action = mocker.patch.object(fedbiomed.node.cli, action)

    node_cli.run("training-plan", *argv)

    run_action.assert_called_once_with(node_cli.plans, **expected)


def test_training_plan_list_shows_the_plans(node_cli):
    node_cli.run("training-plan", "list")

    node_cli.plans.list_training_plans.assert_called_once_with(verbose=True)


@pytest.mark.parametrize(
    "argv, action, expected",
    [
        (["add"], "add_database", {"initialdir": NODE_DATA_PATH}),
        (
            ["add", "--mnist", "/mnist"],
            "add_database",
            {"interactive": False, "path": "/mnist"},
        ),
        (
            ["add", "--mnist"],
            "add_database",
            {"interactive": False, "path": NODE_DATA_PATH},
        ),
        (["delete"], "delete_database", {}),
        (["delete", "--mnist"], "delete_database", {"interactive": False}),
        (["delete", "--all"], "delete_all_database", {}),
    ],
    ids=[
        "add",
        "add_mnist",
        "add_mnist_in_node_data",
        "delete",
        "delete_mnist",
        "delete_all",
    ],
)
def test_dataset_command_runs_its_action(node_cli, mocker, argv, action, expected):
    run_action = mocker.patch.object(fedbiomed.node.cli, action)

    node_cli.run("dataset", *argv)

    run_action.assert_called_once_with(node_cli.datasets, **expected)


@pytest.mark.parametrize("datasets, empty", [([], True), ([{"name": "d"}], False)])
def test_dataset_list_says_when_nothing_is_set_up(node_cli, capsys, datasets, empty):
    node_cli.datasets.list_my_datasets.return_value = datasets

    node_cli.run("dataset", "list")

    node_cli.datasets.list_my_datasets.assert_called_once_with(verbose=True)
    assert ("No data has been set up." in capsys.readouterr().out) is empty


@pytest.fixture
def dataset_file(tmp_path):
    """Writes a dataset description file and returns its path."""

    def write(content):
        path = tmp_path / "dataset.json"
        path.write_text(content)
        return str(path)

    return write


@pytest.mark.parametrize(
    "path, expected_path",
    [
        ("/absolute/data", "/absolute/data"),
        ("relative/data", os.path.join(NODE_ROOT, "relative", "data")),
        ("$FBM_TEST_DATA/data", os.path.join("/env/path", "data")),
    ],
    ids=["absolute", "relative_to_node_root", "environment_variable"],
)
def test_dataset_add_from_file_resolves_the_dataset_path(
    node_cli, mocker, monkeypatch, dataset_file, path, expected_path
):
    monkeypatch.setenv("FBM_TEST_DATA", "/env/path")
    add_database = mocker.patch.object(fedbiomed.node.cli, "add_database")
    description = {
        "path": path,
        "data_type": "csv",
        "description": "d",
        "tags": "#t",
        "name": "n",
    }

    node_cli.run("dataset", "add", "--file", dataset_file(json.dumps(description)))

    add_database.assert_called_once_with(
        node_cli.datasets,
        interactive=False,
        path=expected_path,
        data_type="csv",
        description="d",
        tags="#t",
        name="n",
        dataset_parameters=None,
    )


@pytest.mark.parametrize(
    "content", ["not valid json {{{", None], ids=["invalid", "missing"]
)
def test_dataset_add_from_file_exits_on_an_unreadable_file(
    node_cli, dataset_file, tmp_path, content
):
    path = str(tmp_path / "missing.json") if content is None else dataset_file(content)

    with pytest.raises(SystemExit):
        node_cli.run("dataset", "add", "--file", path)


@pytest.fixture
def node_process_manager(mocker):
    """The `NodeProcessManager` the node CLI builds, mocked."""
    mocker.patch("fedbiomed.node.cli.intro")
    return mocker.patch("fedbiomed.node.cli.NodeProcessManager")


@pytest.mark.parametrize(
    "options, node_args, background, force",
    [
        ([], {}, False, False),
        (["--force"], {}, False, True),
        (
            ["--gpu", "--gpu-num", "2", "--debug"],
            {"gpu": True, "gpu_num": 2, "debug": True},
            False,
            False,
        ),
        (["--gpu-only"], {"gpu": True, "gpu_only": True}, False, False),
        (
            ["--gpu", "--gpu-num", "2", "--gpu-only", "--debug", "--background"],
            {"gpu": True, "gpu_num": 2, "gpu_only": True, "debug": True},
            True,
            False,
        ),
    ],
    ids=["defaults", "force", "gpu_and_debug", "gpu_only_implies_gpu", "background"],
)
def test_start_runs_the_node_with_the_given_options(
    node_cli, node_process_manager, options, node_args, background, force
):
    node_cli.run("start", *options)

    node_process_manager.assert_called_once_with(node_cli.config)
    manager = node_process_manager.return_value
    manager.start.assert_called_once_with(
        node_args={**START_DEFAULTS, **node_args},
        background=background,
        actor=CLI_ACTOR,
        force=force,
    )
    manager.stop.assert_not_called()


def test_start_stops_the_node_on_ctrl_c(node_cli, node_process_manager):
    manager = node_process_manager.return_value
    manager.start.side_effect = KeyboardInterrupt

    with pytest.raises(SystemExit) as exit_:
        node_cli.run("start")

    assert exit_.value.code == 0
    manager.stop.assert_called_once_with(actor=CLI_ACTOR, reason="keyboard_interrupt")


def test_stop_stops_the_node(node_cli, node_process_manager):
    node_cli.run("stop")

    node_process_manager.return_value.stop.assert_called_once_with(
        actor=CLI_ACTOR, reason="cli_stop_command"
    )


@pytest.mark.parametrize(
    "options, node_args, background",
    [
        ([], {}, None),
        (
            ["--gpu", "--gpu-num", "2", "--debug", "--background"],
            {"gpu": True, "gpu_num": 2, "debug": True},
            True,
        ),
        (["--gpu-only"], {"gpu": True, "gpu_only": True}, None),
    ],
    ids=["no_override", "gpu_debug_background", "gpu_only_implies_gpu"],
)
def test_restart_passes_only_the_given_options(
    node_cli, node_process_manager, options, node_args, background
):
    node_cli.run("restart", *options)

    node_process_manager.return_value.restart.assert_called_once_with(
        node_args=node_args,
        background=background,
        actor=CLI_ACTOR,
        reason="cli_restart_command",
    )


@pytest.fixture
def node_status(mocker, node_process_manager):
    """Process and connection state `status` reports: stopped, never connected."""
    process = node_process_manager.return_value
    process.get_status.return_value = NodeState.STOPPED
    connections = mocker.patch("fedbiomed.node.cli.NodeConnectionStateManager")
    connections.return_value.get_connection_state.return_value = None
    connections.return_value.get_connection_history.return_value = []
    return SimpleNamespace(process=process, connections=connections.return_value)


def _connection_entry(**overrides):
    fields = {
        "node_id": "node-1",
        "state": "connected",
        "host": "10.0.0.9",
        "port": "50051",
        "mtls": True,
        "updated_at": "2026-09-03T07:18:32Z",
    }
    return NodeConnectionStateEntry(**{**fields, **overrides})


def test_status_reports_the_recorded_connection(node_cli, node_status, capsys):
    node_status.connections.get_connection_state.return_value = _connection_entry()

    node_cli.run("status")

    out = capsys.readouterr().out
    assert "Node process" in out and "stopped" in out
    assert "10.0.0.9" in out and "50051" in out
    assert "Mutual authentication (mTLS)" in out and "On" in out
    assert "Identity verified" in out and "Unknown" in out
    for field in ("State", "Host", "Port", "Reason", "Connected since", "Last update"):
        assert field in out
    # Nothing is written while the node is stopped, so say the state is not current
    assert "not running" in out


def test_status_without_a_recorded_connection(node_cli, node_status, capsys):
    node_cli.run("status")

    assert "none recorded yet" in capsys.readouterr().out


@pytest.mark.parametrize(
    "state, color", [(NodeState.RUNNING, "\033[32m"), (NodeState.STOPPED, "\033[31m")]
)
@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "not_terminal"])
def test_status_colors_the_process_state_only_on_a_terminal(
    node_cli, node_status, mocker, monkeypatch, capsys, state, color, terminal
):
    node_status.process.get_status.return_value = state
    monkeypatch.delenv("NO_COLOR", raising=False)
    mocker.patch("fedbiomed.node.cli.sys.stdout.isatty", return_value=terminal)

    node_cli.run("status")

    out = capsys.readouterr().out
    assert (f"{color}{state.value}\033[0m" in out) is terminal
    assert state.value in out


@pytest.mark.parametrize("history, listed", [([], False), (["--history"], True)])
def test_status_lists_the_connection_history_only_when_asked(
    node_cli, node_status, capsys, history, listed
):
    entries = [
        _connection_entry(operation="handshake_complete"),
        _connection_entry(state="disconnected", operation="node_starting"),
    ]
    node_status.connections.get_connection_state.return_value = entries[0]
    node_status.connections.get_connection_history.return_value = entries

    node_cli.run("status", *history)

    out = capsys.readouterr().out
    assert ("Recent connection state changes" in out) is listed
    assert ("node_starting" in out) is listed


def test_certificate_check_is_node_only(mocker):
    """A researcher's readiness is a different question, so it has no `check`."""
    mocker.patch("fedbiomed.node.cli.node_component.initiate")
    assert "check" in NodeCLI()._certificate_sub_parsers.choices

    common = CommonCLI()
    common.initialize_certificate_parser()
    assert "check" not in common._certificate_sub_parsers.choices
    # The commands both components share are still there
    assert "register" in common._certificate_sub_parsers.choices


@pytest.mark.parametrize(
    "diagnostics, exit_code, expected",
    [
        (
            [
                CertificateDiagnostic(DiagnosticSeverity.PROBLEM, "nothing registered"),
                CertificateDiagnostic(DiagnosticSeverity.WARNING, "expires soon"),
            ],
            1,
            [
                "1 problem(s) stop this node from starting",
                "nothing registered",
                "expires soon",
            ],
        ),
        (
            [CertificateDiagnostic(DiagnosticSeverity.WARNING, "host not in SAN")],
            None,
            ["1 warning(s)", "host not in SAN"],
        ),
        ([], None, ["No certificate problem found"]),
    ],
    ids=["problem_fails", "warning_passes", "clean"],
)
def test_certificate_check_reports_and_fails_only_on_problems(
    node_cli, mocker, capsys, diagnostics, exit_code, expected
):
    """A non-zero exit says the node would not start, for a deployment script."""
    node_cli.config.getbool.return_value = True
    mocker.patch("fedbiomed.node.cli.certificate_diagnostics", return_value=diagnostics)

    if exit_code is None:
        node_cli.run("certificate", "check")
    else:
        with pytest.raises(SystemExit) as exit_:
            node_cli.run("certificate", "check")
        assert exit_.value.code == exit_code

    out = capsys.readouterr().out
    for text in expected:
        assert text in out
    if exit_code is None:
        assert "problem(s)" not in out


def test_intro_shows_the_active_node(monkeypatch, capsys):
    monkeypatch.setenv("FEDBIOMED_ACTIVE_NODE_ID", "test-node-id")

    intro()

    assert "test-node-id" in capsys.readouterr().out


# ============================================================================
# CLI UTILITIES
# ============================================================================


@pytest.fixture
def medical_folder_cli(mocker):
    """Mocked user input, path prompt and MedicalFolderController class."""
    module = "fedbiomed.node.cli_utils._medical_folder_dataset"
    controller = MagicMock()
    controller.df_dir = {"modality": MagicMock()}
    controller.df_dir["modality"].unique.return_value = ["T1", "T2", "label"]
    controller.demographics_column_names.return_value = ["subject_id", "age", "gender"]
    return SimpleNamespace(
        input=mocker.patch("builtins.input"),
        path=mocker.patch(f"{module}.validated_path_input"),
        controller_class=mocker.patch(
            f"{module}.MedicalFolderController", return_value=controller
        ),
        controller=controller,
    )


@pytest.mark.parametrize(
    "dataset_parameters, dlp",
    [(None, None), ({"custom_param": "value"}, None), (None, DataLoadingPlan())],
    ids=["nothing_given", "parameters_given", "plan_given"],
)
def test_add_medical_folder_dataset_without_demographics_keeps_what_is_given(
    medical_folder_cli, dataset_parameters, dlp
):
    medical_folder_cli.path.return_value = MEDICAL_DATA_PATH
    medical_folder_cli.input.return_value = "n"

    path, parameters, result_dlp = add_medical_folder_dataset_from_cli(
        dataset_parameters=dataset_parameters, dlp=dlp
    )

    assert path == MEDICAL_DATA_PATH
    assert parameters == (dataset_parameters or {})
    assert result_dlp is dlp
    medical_folder_cli.path.assert_called_once_with(type="dir", initialdir=None)
    medical_folder_cli.controller_class.assert_called_once_with(MEDICAL_DATA_PATH)
    medical_folder_cli.input.assert_called_once_with(
        "\nWould you like to select a demographics csv file? [y/N]\n"
    )


@pytest.mark.parametrize(
    "answers, index_col, warned",
    [(["y", "1"], 1, False), (["y", "invalid", "2"], 2, True)],
    ids=["numeric_index", "asks_again_after_non_numeric_index"],
)
def test_add_medical_folder_dataset_with_demographics_asks_the_subject_column(
    medical_folder_cli, recwarn, answers, index_col, warned
):
    medical_folder_cli.path.side_effect = [MEDICAL_DATA_PATH, DEMOGRAPHICS_PATH]
    medical_folder_cli.input.side_effect = answers

    _, parameters, _ = add_medical_folder_dataset_from_cli(
        dataset_parameters={"existing_param": "value"}, dlp=None
    )

    assert parameters == {
        "existing_param": "value",
        "tabular_file": DEMOGRAPHICS_PATH,
        "index_col": index_col,
    }
    medical_folder_cli.controller.demographics_column_names.assert_called_once_with(
        DEMOGRAPHICS_PATH
    )
    assert any("numeric value" in str(w.message) for w in recwarn) is warned


@pytest.mark.parametrize(
    "folders, answers, expected_map, warned",
    [
        (["Should map to T1"], ["1"], {"T1": ["Should map to T1"]}, False),
        (["Should map to T1"], ["wrong", "5", "1"], {"T1": ["Should map to T1"]}, True),
        (
            ["Should map to T1", "Should map to Tnew"],
            ["1", "0", "Tnew", "y", "4"],
            {"T1": ["Should map to T1"], "Tnew": ["Should map to Tnew"]},
            False,
        ),
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
    ids=["one_folder", "wrong_inputs_then_correct", "new_modality", "several_folders"],
)
def test_get_map_modalities2folders_from_cli_maps_each_folder(
    monkeypatch, recwarn, folders, answers, expected_map, warned
):
    answers = iter(answers)
    monkeypatch.setattr(
        "fedbiomed.node.cli_utils._medical_folder_dataset.input",
        lambda _prompt: next(answers),
        raising=False,
    )

    dlb = get_map_modalities2folders_from_cli(folders)

    assert dlb.map == expected_map
    assert any(issubclass(w.category, UserWarning) for w in recwarn) is warned


@pytest.mark.parametrize(
    "action, warning",
    [
        (update_training_plan, "No registered training plans has been found to update"),
        (delete_training_plan, "No training plans to delete"),
    ],
    ids=["update", "delete"],
)
def test_training_plan_action_warns_when_no_plan_applies(mocker, action, warning):
    """Default training plans are files: they can be neither updated nor deleted."""
    logger = mocker.patch("fedbiomed.node.cli_utils._training_plan_management.logger")
    manager = MagicMock()
    manager.list_training_plans.return_value = [
        {"training_plan_type": TrainingPlanStatus.DEFAULT.value}
    ]

    action(manager)

    logger.warning.assert_called_once_with(warning)
    manager.delete_training_plan.assert_not_called()


@pytest.fixture
def cli_database(mocker, monkeypatch, tmp_path):
    """CLI dataset utilities on a mocked manager, with their prompts mocked.

    Path prompts answer `root`, an existing folder; `answers(...)` scripts what
    the user types; `added(*fields)` is what was registered for these fields.
    """
    module = "fedbiomed.node.cli_utils._database"
    manager = MagicMock()
    root = str(tmp_path)

    def answers(*replies):
        replies = iter(replies)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))

    return SimpleNamespace(
        manager=manager,
        added=lambda *fields: {
            field: manager.add_database.call_args.kwargs[field] for field in fields
        },
        root=root,
        answers=answers,
        logger=mocker.patch(f"{module}.logger"),
        data_type=mocker.patch(f"{module}.validated_data_type_input"),
        path=mocker.patch(f"{module}.validated_path_input", return_value=root),
        prompt_path=mocker.patch(f"{module}._prompt_path_cli", return_value=root),
        medical_folder=mocker.patch(f"{module}.add_medical_folder_dataset_from_cli"),
        messagebox=mocker.patch(f"{module}.messagebox"),
    )


@pytest.mark.parametrize(
    "given, registered",
    [
        (
            {**CLI_DATASET, "tags": "tag1,tag2", "data_type": "CSV"},
            {
                "name": "data",
                "tags": ["tag1", "tag2"],
                "description": "desc",
                "data_type": "csv",
            },
        ),
        (
            {**CLI_DATASET, "tags": ["tag1", "tag2"]},
            {
                "name": "data",
                "tags": ["tag1", "tag2"],
                "description": "desc",
                "data_type": "csv",
            },
        ),
        (
            {**CLI_DATASET, "data_type": "unknown"},
            {
                "name": "data",
                "tags": ["tag1"],
                "description": "desc",
                "data_type": "default",
            },
        ),
        (
            {},
            {
                "name": "MNIST",
                "tags": ["#MNIST", "#dataset"],
                "description": "MNIST database",
                "data_type": "default",
            },
        ),
    ],
    ids=[
        "type_lowercased",
        "tags_as_a_list",
        "unknown_type_is_mnist",
        "only_a_path_is_mnist",
    ],
)
def test_add_database_registers_the_given_dataset(cli_database, given, registered):
    add_database(
        cli_database.manager, interactive=False, path=cli_database.root, **given
    )

    cli_database.manager.add_database.assert_called_once_with(
        **registered,
        path=cli_database.root,
        dataset_parameters=None,
        data_loading_plan=None,
    )
    cli_database.manager.list_my_datasets.assert_called_once_with(verbose=True)


@pytest.mark.parametrize(
    "given, refusal",
    [
        ({**CLI_DATASET, "name": "ab"}, "at least 3 characters"),
        ({**CLI_DATASET, "description": "ab"}, "at least 3 characters"),
        ({**CLI_DATASET, "tags": "tag1,ab"}, "at least 3 characters"),
        ({**CLI_DATASET, "tags": ""}, "tags are not empty"),
        ({"path": None}, "path is not set"),
    ],
    ids=["short_name", "short_description", "short_tag", "no_tags", "no_path"],
)
def test_add_database_refuses_incomplete_parameters(cli_database, given, refusal):
    with pytest.raises(FedbiomedDatasetManagerError, match=refusal):
        add_database(
            cli_database.manager,
            interactive=False,
            **{"path": cli_database.root, **given},
        )

    cli_database.manager.add_database.assert_not_called()


def test_add_database_asks_again_for_a_path_that_does_not_exist(cli_database):
    missing = os.path.join(cli_database.root, "missing")

    add_database(cli_database.manager, interactive=False, path=missing, **CLI_DATASET)

    cli_database.prompt_path.assert_called_once_with("csv")
    assert cli_database.added("path") == {"path": cli_database.root}


@pytest.mark.parametrize(
    "answers, tags",
    [
        ([""], ["#MEDNIST", "#dataset"]),
        (["maybe", "y"], ["#MEDNIST", "#dataset"]),
        (["n", "ab", "tag1, tag2"], ["tag1", "tag2"]),
    ],
    ids=["default_tags", "asks_again_on_an_unclear_answer", "own_tags"],
)
def test_add_database_lets_the_user_confirm_predefined_tags(
    cli_database, answers, tags
):
    cli_database.data_type.return_value = "mednist"
    cli_database.answers(*answers)

    add_database(cli_database.manager)

    assert cli_database.added("name", "data_type", "tags") == {
        "name": "MEDNIST",
        "data_type": "mednist",
        "tags": tags,
    }


def test_add_database_asks_until_name_tags_and_description_are_long_enough(
    cli_database,
):
    cli_database.data_type.return_value = "csv"
    cli_database.answers("ab", "data", "ab", "tag1, tag2", "ab", "desc")

    add_database(cli_database.manager)

    assert cli_database.added("name", "tags", "description", "path") == {
        "name": "data",
        "tags": ["tag1", "tag2"],
        "description": "desc",
        "path": cli_database.root,
    }
    cli_database.path.assert_called_once_with("csv", initialdir=None)


def test_add_database_asks_a_custom_dataset_path_until_it_exists(cli_database):
    cli_database.data_type.return_value = "custom"
    missing = os.path.join(cli_database.root, "missing")
    cli_database.answers("data", "tag1", "desc", missing, cli_database.root)

    add_database(cli_database.manager)

    assert cli_database.added("path") == {"path": os.path.realpath(cli_database.root)}


def test_add_database_names_the_medical_folder_loading_plan(cli_database):
    plan = DataLoadingPlan()
    cli_database.data_type.return_value = "medical-folder"
    cli_database.medical_folder.return_value = (
        cli_database.root,
        {"index_col": 0},
        plan,
    )
    # "abc" is refused: plan names have at least 4 characters
    cli_database.answers("data", "tag1", "desc", "abc", "plan")

    add_database(cli_database.manager)

    assert plan.desc == "plan"
    assert cli_database.added("dataset_parameters", "data_loading_plan") == {
        "dataset_parameters": {"index_col": 0},
        "data_loading_plan": plan,
    }


@pytest.mark.parametrize(
    "interactive, error, shown",
    [
        (True, FedbiomedDatasetManagerError("refused"), "refused"),
        (False, FedbiomedDatasetManagerError("refused"), "refused"),
        (False, FedbiomedDatasetError("unreadable"), "Demographic file"),
        (False, FedbiomedError("FB322: conflicting tags"), "FB322"),
    ],
    ids=[
        "refused_in_a_window",
        "refused_in_the_log",
        "unreadable_with_a_hint",
        "manager_refusal_in_the_log",
    ],
)
def test_add_database_exits_when_the_dataset_cannot_be_added(
    cli_database, interactive, error, shown
):
    cli_database.data_type.return_value = "default"
    cli_database.answers("")
    cli_database.manager.add_database.side_effect = error

    with pytest.raises(SystemExit):
        add_database(
            cli_database.manager, interactive=interactive, path=cli_database.root
        )

    report = (
        cli_database.messagebox.showwarning
        if interactive
        else cli_database.logger.error
    )
    assert shown in str(report.call_args)


@pytest.mark.parametrize(
    "datasets, delete, warning",
    [
        ([], delete_database, "No dataset to delete"),
        ([], partial(delete_database, interactive=False), "No dataset to delete"),
        ([], delete_all_database, "No dataset to delete"),
        (
            [{"name": "Other", "dataset_id": "d1"}],
            partial(delete_database, interactive=False),
            "No matching dataset to delete",
        ),
    ],
    ids=["interactive_none", "mnist_none", "all_none", "mnist_not_registered"],
)
def test_delete_with_nothing_to_delete_only_warns(
    cli_database, datasets, delete, warning
):
    cli_database.manager.list_my_datasets.return_value = datasets

    delete(cli_database.manager)

    cli_database.manager.delete_dataset_by_id.assert_not_called()
    cli_database.logger.warning.assert_called_once_with(warning)


@pytest.mark.parametrize(
    "delete, answers, deleted",
    [
        (delete_database, ["x", "9", "2"], ["d2"]),
        (partial(delete_database, interactive=False), [], ["d2"]),
        (delete_all_database, [], ["d1", "d2"]),
    ],
    ids=["chosen_after_invalid_answers", "mnist", "all"],
)
def test_delete_removes_datasets_with_their_derived_ones(
    cli_database, delete, answers, deleted
):
    cli_database.manager.list_my_datasets.return_value = [
        {"name": "Other", "dataset_id": "d1"},
        {"name": "MNIST", "dataset_id": "d2"},
    ]
    cli_database.answers(*answers)

    delete(cli_database.manager)

    assert cli_database.manager.delete_dataset_by_id.call_args_list == [
        call(dataset_id, recursive=True) for dataset_id in deleted
    ]
