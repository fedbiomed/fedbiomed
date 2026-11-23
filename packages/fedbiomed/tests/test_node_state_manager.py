import json
import os

import pytest
from fedbiomed.common.constants import NODE_STATE_PREFIX, __node_state_version__
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import (
    FedbiomedNodeStateManagerError,
    FedbiomedVersionError,
)
from fedbiomed.node.db_models import NodeStateEntry
from fedbiomed.node.node_state_manager import NodeStateFileName, NodeStateManager

_FAKE_DECLEARN_OPTIMIZER_STATE = {
    "config": {
        "lrate": 0.2,
        "w_decay": 0.0,
        "regularizers": [],
        "modules": [
            (
                "adam",
                {"beta_1": 0.9, "beta_2": 0.99, "amsgrad": False, "eps": 1e-07},
            )
        ],
    },
    "states": {
        "modules": [
            (
                "adam",
                {
                    "steps": 0,
                    "vmax": None,
                    "momentum": {"state": 0.0},
                    "velocity": {"state": 0.0},
                },
            )
        ]
    },
}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.json")


@pytest.fixture
def nsm(tmp_path, db_path):
    return NodeStateManager(dir=str(tmp_path), node_id="test-node-id", db_path=db_path)


def _refuse_makedirs(*args, **kwargs):
    """Replaces `os.makedirs`: folders cannot be created."""
    raise PermissionError("raised for the sake of testing")


def test_node_state_manager_get(nsm):
    nsm.initialize()
    state_id = nsm.add("experiment_id", {"optimizer_state": None})

    res = nsm.get("experiment_id", state_id)

    assert res["optimizer_state"] is None
    assert res["state_id"] == state_id


def test_node_state_manager_get_refuses_incompatible_version(nsm, db_path):
    with get_database(db_path).transaction() as session:
        session.add(
            NodeStateEntry(
                state_id="s1",
                experiment_id="experiment_id",
                version_node_id="99.0",
                state={},
            )
        )

    with pytest.raises(FedbiomedVersionError):
        nsm.get("experiment_id", "s1")


def test_node_state_manager_get_error(nsm):
    # no state with this id
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.get("experiment_id", "state_id")

    # a state with this id, but of another experiment
    nsm.initialize()
    state_id = nsm.add("experiment_id", {})
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.get("another_experiment_id", state_id)


def test_node_state_manager_add(nsm):
    experiment_id = "experiment_id"

    # TODO: add states that are framework native (here we are using format that is very similar to Declearn optimizers)

    nsm.initialize()
    res = nsm.add(experiment_id, _FAKE_DECLEARN_OPTIMIZER_STATE)

    # checks: saved as JSON, so tuples come back as lists
    assert res.startswith(NODE_STATE_PREFIX)
    assert nsm.state_id == res
    header = {
        "version_node_id": str(__node_state_version__),
        "state_id": res,
        "experiment_id": experiment_id,
    }
    assert nsm.get(experiment_id, res) == json.loads(
        json.dumps({**_FAKE_DECLEARN_OPTIMIZER_STATE, **header})
    )


def test_node_state_manager_add_refuses_before_initialize(nsm):
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.add(experiment_id="experiment_id", state=_FAKE_DECLEARN_OPTIMIZER_STATE)


def test_node_state_manager_initialize_node_state_manager(tmp_path, nsm):
    nsm.initialize(previous_state_id="previous_state_id")

    expected_path = tmp_path / "var" / "node_state_test-node-id"
    assert expected_path.is_dir()
    assert nsm.previous_state_id == "previous_state_id"
    assert nsm.get_node_state_base_dir() == str(expected_path)


def test_node_state_manager_initialize_node_state_manager_fails(nsm, monkeypatch):
    monkeypatch.setattr(os, "makedirs", _refuse_makedirs)
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.initialize()


def test_node_state_manager_generate_folder_and_create_file_name(nsm):
    # testing values for initializing folders for a given experiment and round
    experiment_id, round_nb = "experiment_id", 4321
    nsm.initialize()

    res = nsm.generate_folder_and_create_file_name(
        experiment_id, round_nb, NodeStateFileName.OPTIMIZER
    )

    experiment_dir = os.path.join(
        nsm.get_node_state_base_dir(), "experiment_id_%s" % experiment_id
    )
    assert os.path.isdir(experiment_dir)
    assert res == os.path.join(
        experiment_dir, NodeStateFileName.OPTIMIZER.value % (round_nb, nsm.state_id)
    )


def test_node_state_manager_generate_folder_and_create_file_name_failures(
    nsm, monkeypatch
):
    experiment_id, round_nb = "experiment_id", 4321

    # raise first exception: case where `node_state_base_dir` has not been found
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.generate_folder_and_create_file_name(
            experiment_id, round_nb, NodeStateFileName.OPTIMIZER
        )

    # raise second exception: the experiment folder cannot be created
    nsm.initialize()
    monkeypatch.setattr(os, "makedirs", _refuse_makedirs)
    with pytest.raises(FedbiomedNodeStateManagerError):
        nsm.generate_folder_and_create_file_name(
            experiment_id, round_nb, NodeStateFileName.OPTIMIZER
        )


@pytest.mark.parametrize("entry_value", NodeStateFileName.list())
def test_node_state_file_name_correct_format_entries(entry_value):
    # here we test that all entries of NodeStateFIleName enum class respect convention
    entry_value % ("string_1", "string_2")


def test_node_state_file_name_model_weights_entry():
    """Test that MODEL_WEIGHTS entry exists and formats correctly."""
    entry = NodeStateFileName.MODEL_WEIGHTS.value

    formatted = entry % ("round_id", "node_state_id")

    assert formatted == "persistent_model_weights_round_id_node_state_id"
