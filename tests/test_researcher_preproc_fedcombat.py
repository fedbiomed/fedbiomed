"""Tests for FedCombatPreproc federated preprocessing workflow."""

import copy
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

import fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat import (
    FedCombatPreproc,
)
from fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat_parameters import (
    _FedCombatParameters,
)
from fedbiomed.researcher.requests import Requests


@pytest.fixture
def mock_fds():
    def _set_federated_dataset(datasets: dict) -> None:
        fds.data.return_value = {
            "n1": {"dataset_id": datasets.get("n1"), "data_type": "csv"},
            "n2": {"dataset_id": datasets.get("n2"), "data_type": "csv"},
        }

    fds = MagicMock(spec=FederatedDataset)
    fds.set_federated_dataset.side_effect = _set_federated_dataset
    _set_federated_dataset({"n1": "initial1", "n2": "initial2"})
    return fds


@pytest.fixture
def mock_reqs():
    return MagicMock(spec=Requests)


@pytest.fixture
def mock_fedcombat_param(monkeypatch: MonkeyPatch):
    mock_param_instance = MagicMock(spec=_FedCombatParameters)
    mock_param_instance.return_value = ({"dict1": 1}, {"dict2": 2})

    monkeypatch.setattr(
        fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat,
        "_FedCombatParameters",
        lambda *args, **kwargs: mock_param_instance,
    )
    return mock_param_instance


@pytest.fixture
def mock_experiment_class():
    return MagicMock(spec=Experiment)


@pytest.fixture
def base_preproc(
    mock_fds, mock_reqs, tmp_path, mock_fedcombat_param, mock_experiment_class
):
    return FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp1",
        researcher_id="r1",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
        preproc_args={
            "covariates": ["cov1", "cov2"],
            "phenotypes": ["phen1"],
        },
    )


def test_check_fds__failed_raises(
    mock_fds, mock_reqs, tmp_path, mock_fedcombat_param, mock_experiment_class
):
    mock_fds.data.side_effect = FedbiomedExperimentError("data retrieval failed")

    with pytest.raises(FedbiomedExperimentError):
        FedCombatPreproc(
            fds=mock_fds,
            experiment_id="e_chk2",
            researcher_id="r_chk",
            reqs=mock_reqs,
            experiment_class=mock_experiment_class,
            nodes=["n1", "n2"],
            experimentation_folder=str(tmp_path),
        )


def test_check_fds_missing_data_type_raises(
    mock_fds, mock_reqs, tmp_path, mock_fedcombat_param, mock_experiment_class
):
    # n2 is missing 'data_type'
    mock_fds.data.return_value = {
        "n1": {"dataset_id": "ds1", "data_type": "csv"},
        "n2": {"dataset_id": "ds2"},
    }

    with pytest.raises(FedbiomedExperimentError):
        FedCombatPreproc(
            fds=mock_fds,
            experiment_id="e_chk2",
            researcher_id="r_chk",
            reqs=mock_reqs,
            experiment_class=mock_experiment_class,
            nodes=["n1", "n2"],
            experimentation_folder=str(tmp_path),
        )


def test_execute_success(monkeypatch, base_preproc, mock_fds, mock_fedcombat_param):
    # patch PreprocRequestJob to a simple callable that returns replies for nodes
    class DummyJob:
        def __init__(self, **kwargs):
            self._nodes = kwargs.get("nodes", [])

        def execute(self):
            return {
                n: MagicMock(
                    preproc_output={
                        f"key_{n}": f"value_{n}",
                        "standardized_dataset": f"std_fds_{n}",
                        "harmonized_dataset": f"ds_{n}",
                    }
                )
                for n in self._nodes
            }

    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat.PreprocRequestJob",
        DummyJob,
    )

    preproc = base_preproc
    assert preproc._needs_harmonization() is True
    assert preproc._harmonized_datasets is None

    result = preproc.execute()
    assert result is True
    assert preproc._needs_harmonization() is False
    assert preproc._harmonized_datasets == {"n1": "ds_n1", "n2": "ds_n2"}


def test_execute_fedcombat_parameters_exception(
    monkeypatch,
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_experiment_class,
):
    # Ensure the FedCombatPreproc class uses a mocked _FedCombatParameters that returns
    # an instance which raises when called: fedcombat_parameters = _FedCombatParameters(...);
    # then fedcombat_parameters(...) triggers the exception.
    bad_fedcombat_parameters_instance = MagicMock()
    bad_fedcombat_parameters_instance.side_effect = FedbiomedExperimentError(
        "fedcombat_parameters failure"
    )

    def bad_fedcombat_parameters_factory(*args, **kwargs):
        # Constructor succeeds and returns an instance that will raise when called
        return bad_fedcombat_parameters_instance

    monkeypatch.setattr(
        fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat,
        "_FedCombatParameters",
        bad_fedcombat_parameters_factory,
    )

    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp_fedcombat_params",
        researcher_id="r_fc",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
        preproc_args={
            "covariates": ["cov1", "cov2"],
            "phenotypes": ["phen1"],
        },
    )

    with pytest.raises(FedbiomedExperimentError) as exc:
        preproc.execute()

    assert "fedcombat_parameters failure" in str(exc.value)


def test_set_nodes_updates_nodes(
    base_preproc, mock_fedcombat_param, mock_experiment_class
):
    preproc = base_preproc
    assert preproc._nodes == ["n1", "n2"]

    preproc.set_nodes(["n3", "n4"])
    assert preproc._nodes == ["n3", "n4"]


def test_execute_not_enough_nodes_raises(
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_fedcombat_param,
    mock_experiment_class,
):
    for nodes in ([], ["n1"]):
        preproc = FedCombatPreproc(
            fds=mock_fds,
            experiment_id="exp2",
            researcher_id="r2",
            reqs=mock_reqs,
            experiment_class=mock_experiment_class,
            nodes=nodes,
            experimentation_folder=str(tmp_path),
        )

        with pytest.raises(FedbiomedExperimentError) as exc:
            preproc.execute()
        assert ErrorNumbers.FB420.value in str(exc.value)


def test_needs_harmonization_true_nodes_changed(
    monkeypatch,
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_fedcombat_param,
    mock_experiment_class,
):
    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp3",
        researcher_id="r3",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
    )
    # simulate previous harmonization with different set of nodes
    preproc._harmonized = True
    preproc._harmonized_datasets = {"n1": "ds1"}

    assert preproc._needs_harmonization() is True

    # simulate previous harmonization with different dataset ids
    preproc._harmonized_datasets = {"n1": "ds1", "n2": "ds_another"}
    assert preproc._needs_harmonization() is True


def test_execute_not_needed_returns_false(
    monkeypatch,
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_fedcombat_param,
    mock_experiment_class,
):
    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp3",
        researcher_id="r3",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1"],
        experimentation_folder=str(tmp_path),
    )
    # simulate previous harmonization with same dataset ids
    preproc._harmonized = True
    preproc._harmonized_datasets = {"n1": "initial1"}

    assert preproc._needs_harmonization() is False
    assert preproc.execute() is False


def test_execute_missing_replies_raises(
    monkeypatch,
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_fedcombat_param,
    mock_experiment_class,
):
    # PreprocRequestJob that returns replies missing nodes
    class BadJob:
        def __init__(self, **kwargs):
            self._nodes = kwargs.get("nodes", [])

        def execute(self):
            return {}

    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat._fedcombat.PreprocRequestJob",
        BadJob,
    )

    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp4",
        researcher_id="r4",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
    )

    with pytest.raises(FedbiomedExperimentError) as exc:
        preproc.execute()
    assert ErrorNumbers.FB420.value in str(exc.value)


def test_save_and_load_state_breakpoint(
    mock_fds,
    mock_reqs,
    tmp_path,
    mock_fedcombat_param,
    mock_experiment_class,
):
    _prepoc_args = {"param1": 10, "param2": "value"}
    _experiment_id = "exp5"

    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id=_experiment_id,
        researcher_id="r5",
        reqs=mock_reqs,
        experiment_class=mock_experiment_class,
        nodes=["n1"],
        experimentation_folder=str(tmp_path),
        preproc_args=_prepoc_args,
    )

    # simulate harmonization done state
    preproc._harmonized = True
    preproc._harmonized_datasets = {"n1": "ds1"}

    state = preproc.save_state_breakpoint()
    # Build a full state for load_state_breakpoint: include required init args
    full_state = copy.deepcopy(state)
    full_state["arguments"].update(
        {
            "fds": mock_fds,
            "experiment_id": _experiment_id,
            "researcher_id": "r5",
            "reqs": mock_reqs,
            "experiment_class": mock_experiment_class,
            "nodes": ["n1"],
            "experimentation_folder": str(tmp_path),
        }
    )

    loaded = FedCombatPreproc.load_state_breakpoint(full_state)

    assert loaded._preproc_args == preproc._preproc_args
    assert loaded._preproc_id == preproc._preproc_id
    assert loaded._harmonized == preproc._harmonized
    assert loaded._harmonized_datasets == preproc._harmonized_datasets
