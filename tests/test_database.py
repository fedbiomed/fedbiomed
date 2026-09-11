from unittest.mock import patch

import pytest

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import (
    DatasetEntry,
    DynamicDatasetEntry,
    NodeProcessStateEntry,
)
from fedbiomed.node.dataset_manager._db_tables import (
    DatasetTable,
    DlpTable,
    DynamicDatasetTable,
    NodeConnectionStateHistoryTable,
    NodeConnectionStateTable,
    NodeProcessStateHistoryTable,
    NodeProcessStateTable,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.json")


@pytest.fixture
def dataset_table(db_path):
    return DatasetTable(db_path)


@pytest.fixture
def dynamic_dataset_table(db_path):
    return DynamicDatasetTable(db_path)


@pytest.fixture
def dlp_table(db_path):
    return DlpTable(db_path)


@pytest.fixture
def process_state_table(db_path):
    return NodeProcessStateTable(db_path)


@pytest.fixture
def connection_state_table(db_path):
    return NodeConnectionStateTable(db_path)


@pytest.fixture
def connection_history_table(db_path):
    return NodeConnectionStateHistoryTable(db_path)


def _dataset(**kwargs):
    return {
        "name": "test_dataset",
        "data_type": "image",
        "tags": ["tag1", "tag2"],
        "description": "A test dataset",
        "path": "/path/to/data",
        "shape": [100, 100],
        "dtypes": {"data": "float32"},
        **kwargs,
    }


def _dynamic_dataset(**kwargs):
    return {
        "path": "/path/to/dynamic",
        "researcher_id": "res_123",
        "experiment_id": "exp_456",
        "processing_id": "proc_789",
        "parent_dataset_id": "dataset_000",
        **kwargs,
    }


def _dlp(**kwargs):
    return {
        "dlp_id": "dlp_123",
        "dlp_name": "valid_dlp",
        "target_dataset_type": DatasetTypes.TABULAR.value,
        "key_paths": "/path/to/plan",
        "loading_blocks": {"block1": "block_id1"},
        **kwargs,
    }


def _process_state(**kwargs):
    return {
        "node_id": "node-1",
        "node_name": "Node 1",
        "pid": 111,
        "state": "running",
        "action": "start",
        **kwargs,
    }


def _connection_state(**kwargs):
    return {
        "node_id": "node-1",
        "state": "failed",
        "host": "localhost",
        "port": "50051",
        "reason": "handshake failed",
        "certificate": {"cert_subject": "CN=RESEARCHER_1", "cert_serial": "1a2b"},
        "updated_at": "2026-08-01T10:00:00Z",
        **kwargs,
    }


@pytest.mark.parametrize(
    "entry_class, fields, prefix",
    [
        (DatasetEntry, _dataset(), "dataset_"),
        (DynamicDatasetEntry, _dynamic_dataset(), "dynamic_dataset_"),
    ],
)
def test_entry_generates_dataset_id_unless_given(entry_class, fields, prefix):
    assert entry_class(**fields).dataset_id.startswith(prefix)
    assert entry_class(**fields, dataset_id="given_id").dataset_id == "given_id"


def test_entry_to_dict_drops_unset_fields():
    entry = DynamicDatasetEntry(**_dynamic_dataset())

    dict_rep = entry.to_dict()

    assert dict_rep["researcher_id"] == "res_123"
    # Unset optional fields are dropped rather than stored as null
    assert "name" not in dict_rep


def test_entry_from_dict_rejects_invalid_data():
    with pytest.raises(FedbiomedError):
        DynamicDatasetEntry.from_dict(_dynamic_dataset(unknown_field="value"))


def test_table_insert_rejects_entry_missing_required_fields(dynamic_dataset_table):
    with pytest.raises(FedbiomedError):
        dynamic_dataset_table.insert({"path": "/path/to/dynamic"})


def test_table_update_by_id_rejects_id_change_and_unknown_entry(dataset_table):
    dataset_id = dataset_table.insert(_dataset())

    with pytest.raises(FedbiomedError):
        dataset_table.update_by_id(dataset_id, {"dataset_id": "another_id"})

    with pytest.raises(FedbiomedError):
        dataset_table.update_by_id("NON_EXISTENT", {"name": "renamed"})


def test_dataset_table_round_trip(dataset_table):
    dataset_id = dataset_table.insert(_dataset(name="dataset_get", tags=["findme"]))

    assert dataset_table.get_by_id(dataset_id)["name"] == "dataset_get"
    assert dataset_table.get_validated_entry(dataset_id).name == "dataset_get"
    assert dataset_table.get_by_id("NON_EXISTENT") is None
    assert [d["name"] for d in dataset_table.search_by_tags(["findme"])] == [
        "dataset_get"
    ]


def test_dataset_table_insert_rejects_conflicting_tags(dataset_table):
    dataset_table.insert(_dataset(tags=["tag1", "tag2"]))

    with pytest.raises(FedbiomedError):
        dataset_table.insert(_dataset(name="conflict_dataset", tags=["tag2"]))


def test_dataset_table_update_by_id_rejects_conflicting_tags(dataset_table):
    dataset_id = dataset_table.insert(
        _dataset(name="dataset_to_update", tags=["special"])
    )
    dataset_table.insert(_dataset(name="other_dataset", tags=["other"]))

    with pytest.raises(FedbiomedError):
        dataset_table.update_by_id(dataset_id, {"tags": ["other", "special"]})

    dataset_table.update_by_id(dataset_id, {"tags": ["unique"]})
    assert dataset_table.get_by_id(dataset_id)["tags"] == ["unique"]


def test_dynamic_dataset_table_round_trip(dynamic_dataset_table):
    entry = _dynamic_dataset(name="Dynamic Dataset 1")

    dataset_id = dynamic_dataset_table.insert(entry)

    assert dynamic_dataset_table.get_by_id(dataset_id) == {
        **entry,
        "dataset_id": dataset_id,
    }
    assert dynamic_dataset_table.get_by_id("NON_EXISTENT") is None


def test_dynamic_dataset_table_collect_subtree(dynamic_dataset_table):
    child_id = dynamic_dataset_table.insert(
        _dynamic_dataset(path="/child", parent_dataset_id="dataset_root_id")
    )
    grandchild_id = dynamic_dataset_table.insert(
        _dynamic_dataset(path="/grandchild", parent_dataset_id=child_id)
    )

    assert dynamic_dataset_table.collect_subtree("dataset_root_id") == [
        child_id,
        grandchild_id,
    ]
    assert dynamic_dataset_table.collect_subtree("unknown_id") == []


@pytest.mark.parametrize(
    "invalid_entry",
    [
        _dlp(target_dataset_type="not_a_type"),
        _dlp(dlp_name="abc"),
    ],
    ids=["unknown target type", "name too short"],
)
def test_dlp_table_insert_rejects_invalid_entry(dlp_table, invalid_entry):
    with pytest.raises(FedbiomedError):
        dlp_table.insert(invalid_entry)


def test_dlp_table_insert_rejects_non_unique_name(dlp_table):
    dlp_table.insert(_dlp(dlp_id="dlp_125", dlp_name="unique_name"))

    with pytest.raises(FedbiomedError):
        dlp_table.insert(
            _dlp(
                dlp_id="dlp_999",
                dlp_name="unique_name",
                target_dataset_type=DatasetTypes.MEDNIST.value,
            )
        )


def test_dlp_table_round_trip_and_list_by_target_dataset_type(dlp_table):
    dlp_table.insert(_dlp(dlp_id="dlp_100", dlp_name="dlp_listed"))

    stored = dlp_table.get_by_id("dlp_100")
    assert stored["dlp_name"] == "dlp_listed"
    assert stored["loading_blocks"] == {"block1": "block_id1"}

    listed = dlp_table.list_by_target_dataset_type(DatasetTypes.TABULAR.value)
    assert [entry["dlp_id"] for entry in listed] == ["dlp_100"]

    with pytest.raises(FedbiomedError):
        dlp_table.list_by_target_dataset_type("invalid")


def test_process_state_table_upserts_and_merges_by_node_id(process_state_table):
    node_args = {"gpu": True, "gpu_num": 2, "gpu_only": False, "debug": True}
    process_state_table.update_or_insert_by_id(
        "node-1", _process_state(node_args=node_args, background=True)
    )

    process_state_table.update_or_insert_by_id(
        "node-1", _process_state(pid=222, state="stopped", action="stop")
    )

    stored = process_state_table.get_by_id("node-1")
    assert (stored["pid"], stored["state"]) == (222, "stopped")
    assert len(process_state_table.all()) == 1
    # Merged, not replaced: settings the second write omits are kept
    assert stored["node_args"] == node_args
    assert stored["background"] is True


def test_process_state_table_rejects_another_node_id(process_state_table):
    with pytest.raises(FedbiomedError):
        process_state_table.update_or_insert_by_id("node-2", _process_state())


def test_process_state_entry_accepts_legacy_records():
    """Records written before execution settings were stored still load."""
    entry = NodeProcessStateEntry.from_dict(_process_state())

    assert entry.node_args is None
    assert entry.background is None


def test_connection_state_table_replaces_by_node_id(connection_state_table):
    connection_state_table.replace_by_id("node-1", _connection_state())

    stored = connection_state_table.get_by_id("node-1")
    # Audit fields are kept whatever they are, so the GUI can show them as they come
    assert stored["certificate"] == {
        "cert_subject": "CN=RESEARCHER_1",
        "cert_serial": "1a2b",
    }

    connection_state_table.replace_by_id(
        "node-1",
        {
            "node_id": "node-1",
            "state": "connected",
            "host": "localhost",
            "port": "50051",
            "identity_verified": False,
            "updated_at": "2026-08-20T10:00:00Z",
        },
    )

    stored = connection_state_table.get_by_id("node-1")
    assert stored["state"] == "connected"
    assert stored["identity_verified"] is False
    assert len(connection_state_table.all()) == 1
    # Replaced, not merged: the previous state's fields are gone
    assert "reason" not in stored
    assert "certificate" not in stored


def test_connection_state_table_rejects_another_node_id(connection_state_table):
    with pytest.raises(FedbiomedError):
        connection_state_table.replace_by_id("node-2", _connection_state())


def test_connection_history_table_appends_entries_for_same_node(
    connection_history_table,
):
    connection_history_table.insert(_connection_state())
    connection_history_table.insert(_connection_state(state="connected"))

    stored = connection_history_table.get_all_by_value("node_id", "node-1")
    assert [entry["state"] for entry in stored] == ["failed", "connected"]


@pytest.mark.parametrize(
    "history_table_class, entry",
    [
        (NodeProcessStateHistoryTable, _process_state()),
        (NodeConnectionStateHistoryTable, _connection_state()),
    ],
    ids=["process state", "connection state"],
)
def test_history_table_deletes_entries_older_than_cutoff(
    db_path, history_table_class, entry
):
    history_table = history_table_class(db_path)
    for updated_at in ("2026-06-01T10:00:00Z", "2026-08-20T10:00:00Z"):
        history_table.insert({**entry, "updated_at": updated_at})

    history_table.delete_older_than("2026-07-21T10:00:00Z")

    assert [entry["updated_at"] for entry in history_table.all()] == [
        "2026-08-20T10:00:00Z"
    ]


def test_history_table_does_not_remove_when_nothing_is_stale(connection_history_table):
    """TinyDB rewrites the whole database file on a removal, matching or not."""
    connection_history_table.insert(_connection_state())

    with patch.object(connection_history_table._table, "remove") as remove:
        removed = connection_history_table.delete_older_than("2020-01-01T00:00:00Z")

    remove.assert_not_called()
    assert removed == []
