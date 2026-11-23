from datetime import datetime, timezone

import fedbiomed.common.db as db_mod
import pytest
from fedbiomed.common.db import get_database
from fedbiomed.common.db_models import Certificate, SecaggElement
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.db_models import (
    Channel,
    Dataset,
    Dlb,
    Dlp,
    DynamicDataset,
    NodeConnectionState,
    NodeConnectionStateHistory,
    NodeProcessState,
    NodeProcessStateHistory,
    NodeState,
    TrainingPlan,
)
from sqlalchemy import delete, inspect, select


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.sqlite")


@pytest.fixture
def database(db_path):
    database = get_database(db_path)
    database.create_tables([Dataset, DynamicDataset, Dlp, Dlb])
    return database


def _dataset(**kwargs):
    return {
        "name": "n",
        "data_type": "csv",
        "tags": ["t"],
        "description": "d",
        "path": "/data",
        "shape": {"csv": [10, 2]},
        "dtypes": {"c": "int"},
        **kwargs,
    }


def _dynamic_dataset(**kwargs):
    return {
        "path": "/data/dyn",
        "researcher_id": "r",
        "experiment_id": "e",
        "processing_id": "p",
        "parent_dataset_id": "dataset_1",
        **kwargs,
    }


def _dlp(**kwargs):
    return {
        "dlp_id": "dlp_1",
        "dlp_name": "plan",
        "target_dataset_type": "medical-folder",
        "loading_blocks": {"key": "dlb_1"},
        "key_paths": {"key": ["a", "b"]},
        **kwargs,
    }


def _dlb(**kwargs):
    return {
        "dlb_id": "dlb_1",
        "content": {
            "dlb_id": "dlb_1",
            "loading_block_class": "Block",
            "loading_block_module": "module",
            "map": {"k": ["v"]},
        },
        **kwargs,
    }


def _add(database, *entries):
    with database.transaction() as session:
        session.add_all(entries)


def _parent():
    """The registered dataset `_dynamic_dataset` derives from."""
    return Dataset.from_dict(_dataset(dataset_id="dataset_1"))


def _child():
    return DynamicDataset.from_dict(_dynamic_dataset(dataset_id="child"))


def _grandchild():
    """Derived from `_child`, a dynamic dataset."""
    return DynamicDataset.from_dict(
        _dynamic_dataset(
            dataset_id="grandchild", parent_dataset_id=None, parent_dynamic_id="child"
        )
    )


@pytest.mark.parametrize(
    "model, data, prefix",
    [
        (Dataset, _dataset(), "dataset_"),
        (DynamicDataset, _dynamic_dataset(), "dynamic_dataset_"),
    ],
)
def test_dataset_id_is_generated_unless_given(model, data, prefix):
    assert model.from_dict(data).dataset_id.startswith(prefix)
    assert model.from_dict({**data, "dataset_id": "given"}).dataset_id == "given"


@pytest.mark.parametrize(
    "model, entry_id, data",
    [
        (
            Dataset,
            "d1",
            _dataset(dataset_id="d1", shape={"m": [1]}, dataset_parameters={"a": [1]}),
        ),
        (DynamicDataset, "d1", _dynamic_dataset(dataset_id="d1")),
        (Dlp, "dlp_1", _dlp()),
        (Dlb, "dlb_1", _dlb()),
    ],
)
def test_entry_round_trip(database, model, entry_id, data):
    _add(database, _parent(), model.from_dict(data))
    with database.transaction() as session:
        assert session.get(model, entry_id).to_dict() == data


@pytest.mark.parametrize(
    "model, data",
    [
        (Dataset, _dataset(data_type="tabular")),
        (DynamicDataset, _dynamic_dataset(data_type="tabular")),
        (Dlp, _dlp(target_dataset_type="tabular")),
    ],
)
def test_dataset_type_must_be_a_dataset_types_value(database, model, data):
    with pytest.raises(FedbiomedError, match="FB634.*CHECK constraint failed"):
        _add(database, _parent(), model.from_dict(data))


def test_dynamic_dataset_accepts_unset_data_type(database):
    _add(database, _parent(), DynamicDataset.from_dict(_dynamic_dataset()))


def test_custom_dataset_has_no_shape_nor_dtypes(database):
    data = _dataset(dataset_id="d1", data_type="custom", shape=None, dtypes=None)
    _add(database, Dataset.from_dict(data))
    with database.transaction() as session:
        assert "shape" not in session.get(Dataset, "d1").to_dict()


@pytest.mark.parametrize("tags", ["t", [1]])
def test_tags_must_be_a_list_of_strings(tags):
    with pytest.raises(FedbiomedError, match="FB632.*Dataset.tags: expected a list"):
        Dataset.from_dict(_dataset(tags=tags))


def test_tags_are_checked_when_assigned(database):
    _add(database, _parent())
    with pytest.raises(FedbiomedError, match="FB632.*Dataset.tags: expected a list"):
        with database.transaction() as session:
            session.get(Dataset, "dataset_1").tags = 5


def test_dlp_name_needs_four_characters(database):
    with pytest.raises(FedbiomedError, match="FB316.*at least 4 characters"):
        _add(database, Dlp.from_dict(_dlp(dlp_name="abc")))


def test_dlp_name_is_unique(database):
    _add(database, Dlp.from_dict(_dlp()))
    with pytest.raises(
        FedbiomedError, match="FB634.*UNIQUE constraint failed: Dlps.dlp_name"
    ):
        _add(database, Dlp.from_dict(_dlp(dlp_id="dlp_2")))


def test_dataset_and_its_dlp_are_saved_in_one_transaction(database):
    _add(database, Dataset.from_dict(_dataset(dlp_id="dlp_1")), Dlp.from_dict(_dlp()))


def test_dataset_refuses_unknown_dlp(database):
    with pytest.raises(FedbiomedError, match="FB634.*FOREIGN KEY constraint failed"):
        _add(database, Dataset.from_dict(_dataset(dlp_id="unknown")))


def test_dlp_in_use_cannot_be_deleted(database):
    _add(database, Dlp.from_dict(_dlp()), Dataset.from_dict(_dataset(dlp_id="dlp_1")))
    with pytest.raises(FedbiomedError, match="FB634.*FOREIGN KEY constraint failed"):
        with database.transaction() as session:
            session.execute(delete(Dlp).where(Dlp.dlp_id == "dlp_1"))


def test_dlp_not_in_use_can_be_deleted(database):
    _add(database, Dlp.from_dict(_dlp()))
    with database.transaction() as session:
        session.delete(session.get(Dlp, "dlp_1"))
    with database.transaction() as session:
        assert session.get(Dlp, "dlp_1") is None


def test_dlp_loaded_from_its_dataset_is_logged_as_read(database, monkeypatch):
    _add(database, Dlp.from_dict(_dlp()), Dataset.from_dict(_dataset(dlp_id="dlp_1")))
    tables = []
    monkeypatch.setattr(
        db_mod.logger,
        "security_event",
        lambda **fields: tables.append((fields["operation"], fields["table"])),
    )

    with database.transaction() as session:
        dataset = session.scalars(select(Dataset)).one()
        assert dataset.dlp.dlp_name == "plan"

    assert tables == [("select", "Datasets"), ("select", "Dlps")]


def test_dynamic_dataset_parents_are_indexed(database):
    indexes = inspect(database._engine).get_indexes("DynamicDatasets")
    assert sorted(index["column_names"] for index in indexes) == [
        ["parent_dataset_id"],
        ["parent_dynamic_id"],
    ]


def test_parent_and_derived_datasets_are_saved_in_one_transaction(database):
    # The registered parent is inserted first even when added last; derived
    # datasets are inserted in the order they are added
    _add(database, _child(), _grandchild(), _parent())


@pytest.mark.parametrize(
    "parents, refusal",
    [
        ({"parent_dataset_id": "unknown"}, "FOREIGN KEY constraint failed"),
        (
            {"parent_dataset_id": None, "parent_dynamic_id": "unknown"},
            "FOREIGN KEY constraint failed",
        ),
        ({"parent_dynamic_id": "child"}, "CHECK constraint failed"),
        ({"parent_dataset_id": None}, "CHECK constraint failed"),
    ],
    ids=["unknown dataset", "unknown dynamic dataset", "two parents", "no parent"],
)
def test_dynamic_dataset_needs_exactly_one_existing_parent(database, parents, refusal):
    _add(database, _parent(), _child())

    with pytest.raises(FedbiomedError, match=f"FB634.*{refusal}"):
        _add(database, DynamicDataset.from_dict(_dynamic_dataset(**parents)))


@pytest.mark.parametrize(
    "model, dataset_id",
    [(Dataset, "dataset_1"), (DynamicDataset, "child")],
)
def test_dataset_with_derived_datasets_cannot_be_deleted(database, model, dataset_id):
    _add(database, _parent(), _child(), _grandchild())

    with pytest.raises(FedbiomedError, match="FB634.*FOREIGN KEY constraint failed"):
        with database.transaction() as session:
            session.execute(delete(model).where(model.dataset_id == dataset_id))


# ------------------------------- tables not yet used by a manager -------------------------------

_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_PROCESS = {"node_name": "n", "state": "running", "action": "start", "pid": 1}
_CONNECTION = {"state": "connected", "host": "h", "port": "50051"}


@pytest.mark.parametrize(
    "model, data",
    [
        (Certificate, {"component_id": "c", "certificate": "PEM"}),
        (
            SecaggElement,
            {
                "secagg_id": "s",
                "secagg_version": "1",
                "secagg_elem": 0,
                "experiment_id": "e",
                "parties": ["r", "n1"],
                "context": {"server_key": 1},
            },
        ),
        (
            Channel,
            {"distant_node_id": "n2", "n2n_channel_version": "1", "local_key": "k"},
        ),
        (
            TrainingPlan,
            {
                "training_plan_id": "tp",
                "name": "plan",
                "hash": "h",
                "algorithm": "SHA256",
                "training_plan": "code",
                "training_plan_type": "registered",
                "training_plan_status": "Approved",
                "date_created": _NOW,
                "date_modified": _NOW,
                "date_registered": _NOW,
            },
        ),
        (
            NodeState,
            {
                "state_id": "s",
                "experiment_id": "e",
                "version_node_id": "1",
                "state": {"optimizer_state": None, "testing_dataset": [1, 2]},
            },
        ),
        (NodeProcessState, {"node_id": "n1", **_PROCESS, "started_at": _NOW}),
        (NodeConnectionState, {"node_id": "n1", **_CONNECTION, "updated_at": _NOW}),
    ],
)
def test_tables_round_trip(db_path, model, data):
    database = get_database(db_path)
    database.create_tables([model])
    _add(database, model.from_dict(data))

    with database.transaction() as session:
        (saved,) = session.scalars(select(model)).all()
        assert saved.to_dict() == data


@pytest.mark.parametrize(
    "model, fields",
    [(NodeProcessStateHistory, _PROCESS), (NodeConnectionStateHistory, _CONNECTION)],
)
def test_history_keeps_several_entries_per_node(db_path, model, fields):
    database = get_database(db_path)
    database.create_tables([model])
    _add(database, model(node_id="n1", **fields), model(node_id="n1", **fields))

    with database.transaction() as session:
        assert len(session.scalars(select(model)).all()) == 2
