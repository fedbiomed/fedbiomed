"""Tests for FedCombatPreproc federated preprocessing workflow."""

import copy
from unittest.mock import MagicMock

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows.preproc._fedcombat import (
    FedCombatPreproc,
)
from fedbiomed.researcher.requests import Requests


@pytest.fixture
def mock_fds():
    fds = MagicMock(spec=FederatedDataset)
    fds.data.return_value = {"n1": {"dataset_id": "ds1"}, "n2": {"dataset_id": "ds2"}}
    fds.node_ids.return_value = ["n1", "n2"]
    return fds


@pytest.fixture
def mock_reqs():
    return MagicMock(spec=Requests)


@pytest.fixture
def base_preproc(mock_fds, mock_reqs, tmp_path):
    return FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp1",
        researcher_id="r1",
        reqs=mock_reqs,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
    )


def test_execute_success(monkeypatch, base_preproc, mock_fds):
    # patch PreprocRequestJob to a simple callable that returns replies for nodes
    class DummyJob:
        def __init__(self, **kwargs):
            self._nodes = kwargs.get("nodes", [])

        def execute(self):
            return {n: f"r_{n}" for n in self._nodes}

    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat.PreprocRequestJob",
        DummyJob,
    )

    preproc = base_preproc
    assert preproc._needs_harmonization() is True
    assert preproc._harmonized_datasets is None

    result = preproc.execute()
    assert result is True
    assert preproc._needs_harmonization() is False
    assert preproc._harmonized_datasets == {"n1": "ds1", "n2": "ds2"}


def test_execute_empty_nodes_raises(mock_fds, mock_reqs, tmp_path):
    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp2",
        researcher_id="r2",
        reqs=mock_reqs,
        nodes=[],
        experimentation_folder=str(tmp_path),
    )

    with pytest.raises(FedbiomedExperimentError) as exc:
        preproc.execute()
    assert ErrorNumbers.FB420.value in str(exc.value)


def test_needs_harmonization_true_nodes_changed(
    monkeypatch, mock_fds, mock_reqs, tmp_path
):
    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp3",
        researcher_id="r3",
        reqs=mock_reqs,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
    )
    # simulate previous harmonization with different set of nodes
    preproc._do_harmonization = False
    preproc._harmonized_datasets = {"n1": "ds1"}

    assert preproc._needs_harmonization() is True

    # simulate previous harmonization with different dataset ids
    preproc._harmonized_datasets = {"n1": "ds1", "n2": "ds_another"}
    assert preproc._needs_harmonization() is True


def test_execute_not_needed_returns_false(monkeypatch, mock_fds, mock_reqs, tmp_path):
    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp3",
        researcher_id="r3",
        reqs=mock_reqs,
        nodes=["n1"],
        experimentation_folder=str(tmp_path),
    )
    # simulate previous harmonization with same dataset ids
    preproc._do_harmonization = False
    preproc._harmonized_datasets = {"n1": "ds1"}

    assert preproc._needs_harmonization() is False
    assert preproc.execute() is False


def test_execute_missing_replies_raises(monkeypatch, mock_fds, mock_reqs, tmp_path):
    # PreprocRequestJob that returns replies missing nodes
    class BadJob:
        def __init__(self, **kwargs):
            self._nodes = kwargs.get("nodes", [])

        def execute(self):
            return {}

    monkeypatch.setattr(
        "fedbiomed.researcher.federated_workflows.preproc._fedcombat.PreprocRequestJob",
        BadJob,
    )

    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id="exp4",
        researcher_id="r4",
        reqs=mock_reqs,
        nodes=["n1", "n2"],
        experimentation_folder=str(tmp_path),
    )

    with pytest.raises(FedbiomedExperimentError) as exc:
        preproc.execute()
    assert ErrorNumbers.FB420.value in str(exc.value)


def test_save_and_load_state_breakpoint(mock_fds, mock_reqs, tmp_path):
    _prepoc_args = {"param1": 10, "param2": "value"}
    _experiment_id = "exp5"

    preproc = FedCombatPreproc(
        fds=mock_fds,
        experiment_id=_experiment_id,
        researcher_id="r5",
        reqs=mock_reqs,
        nodes=["n1"],
        experimentation_folder=str(tmp_path),
        preproc_args=_prepoc_args,
    )

    # simulate harmonization done state
    preproc._do_harmonization = False
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
            "nodes": ["n1"],
            "experimentation_folder": str(tmp_path),
        }
    )

    loaded = FedCombatPreproc.load_state_breakpoint(full_state)

    assert loaded._preproc_args == preproc._preproc_args
    assert loaded._preproc_id == preproc._preproc_id
    assert loaded._do_harmonization == preproc._do_harmonization
    assert loaded._harmonized_datasets == preproc._harmonized_datasets
