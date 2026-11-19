from unittest.mock import patch

import pytest
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import NodeProcessStateEntry
from fedbiomed.node.dataset_manager._db_tables import (
    NodeConnectionStateHistoryTable,
    NodeConnectionStateTable,
    NodeProcessStateHistoryTable,
    NodeProcessStateTable,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.json")


@pytest.fixture
def process_state_table(db_path):
    return NodeProcessStateTable(db_path)


@pytest.fixture
def connection_state_table(db_path):
    return NodeConnectionStateTable(db_path)


@pytest.fixture
def connection_history_table(db_path):
    return NodeConnectionStateHistoryTable(db_path)


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


def test_entry_to_dict_drops_unset_fields():
    entry = NodeProcessStateEntry(**_process_state())

    dict_rep = entry.to_dict()

    assert dict_rep["node_name"] == "Node 1"
    # Unset optional fields are dropped rather than stored as null
    assert "reason" not in dict_rep


def test_entry_from_dict_rejects_invalid_data():
    with pytest.raises(FedbiomedError):
        NodeProcessStateEntry.from_dict(_process_state(unknown_field="value"))


def test_table_insert_rejects_entry_missing_required_fields(connection_history_table):
    with pytest.raises(FedbiomedError):
        connection_history_table.insert({"node_id": "node-1"})


def test_table_update_by_id_rejects_id_change_and_unknown_entry(process_state_table):
    process_state_table.update_or_insert_by_id("node-1", _process_state())

    with pytest.raises(FedbiomedError):
        process_state_table.update_by_id("node-1", {"node_id": "another_id"})

    with pytest.raises(FedbiomedError):
        process_state_table.update_by_id("NON_EXISTENT", {"state": "stopped"})


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
    assert len(process_state_table._table.all()) == 1
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
    assert len(connection_state_table._table.all()) == 1
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

    assert [entry["updated_at"] for entry in history_table._table.all()] == [
        "2026-08-20T10:00:00Z"
    ]


def test_history_table_does_not_remove_when_nothing_is_stale(connection_history_table):
    """TinyDB rewrites the whole database file on a removal, matching or not."""
    connection_history_table.insert(_connection_state())

    with patch.object(connection_history_table._table, "remove") as remove:
        removed = connection_history_table.delete_older_than("2020-01-01T00:00:00Z")

    remove.assert_not_called()
    assert removed == []
