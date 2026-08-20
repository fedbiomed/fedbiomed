import tempfile
import unittest
from unittest.mock import patch

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import (
    DatasetEntry,
    DynamicDatasetEntry,
    NodeConnectionStateEntry,
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


class TestDatasetEntry(unittest.TestCase):
    def test_dataclass_initialization(self):
        entry = DatasetEntry(
            name="Test Dataset",
            data_type="image",
            tags=["tag1"],
            description="A test dataset",
            path="/path/to/dataset",
            shape=[100, 100],
            dtypes={"data": "float32"},
        )
        self.assertEqual(entry.name, "Test Dataset")
        self.assertEqual(entry.data_type, "image")

    def test_dataclass_todict(self):
        entry = DatasetEntry(
            name="Test Dataset",
            data_type="image",
            tags=["tag1"],
            description="A test dataset",
            path="/path/to/dataset",
            shape=[100, 100],
            dtypes={"data": "float32"},
        )
        dict_rep = entry.to_dict()
        self.assertIn("name", dict_rep)
        self.assertIn("data_type", dict_rep)
        self.assertEqual(dict_rep["name"], "Test Dataset")


class TestDynamicDatasetEntry(unittest.TestCase):
    def test_dataclass_initialization(self):
        entry = DynamicDatasetEntry(
            path="/path/to/dynamic",
            researcher_id="res_123",
            experiment_id="exp_456",
            processing_id="proc_789",
            parent_dataset_id="dataset_000",
            name="Dynamic Dataset",
        )
        self.assertEqual(entry.path, "/path/to/dynamic")
        self.assertEqual(entry.researcher_id, "res_123")
        self.assertEqual(entry.experiment_id, "exp_456")
        self.assertEqual(entry.name, "Dynamic Dataset")
        self.assertTrue(entry.dataset_id.startswith("dynamic_dataset_"))

    def test_preserve_given_dataset_id(self):
        entry = DynamicDatasetEntry(
            path="/path/to/dynamic",
            researcher_id="res_123",
            experiment_id="exp_456",
            processing_id="proc_789",
            parent_dataset_id="dataset_000",
            dataset_id="custom_dynamic_id",
        )
        self.assertEqual(entry.dataset_id, "custom_dynamic_id")

    def test_dataclass_todict(self):
        entry = DynamicDatasetEntry(
            path="/path/to/dynamic",
            researcher_id="res_123",
            experiment_id="exp_456",
            processing_id="proc_789",
            parent_dataset_id="dataset_000",
        )
        dict_rep = entry.to_dict()
        self.assertIn("researcher_id", dict_rep)
        self.assertIn("experiment_id", dict_rep)
        self.assertEqual(dict_rep["researcher_id"], "res_123")
        self.assertNotIn("name", dict_rep)

    def test_from_dict(self):
        dict_data = {
            "path": "/path/to/dynamic",
            "researcher_id": "res_123",
            "experiment_id": "exp_456",
            "processing_id": "proc_789",
            "parent_dataset_id": "dataset_000",
            "name": "Dynamic Dataset",
        }
        entry = DynamicDatasetEntry.from_dict(dict_data)
        self.assertEqual(entry.path, "/path/to/dynamic")
        self.assertEqual(entry.researcher_id, "res_123")
        self.assertEqual(entry.experiment_id, "exp_456")
        self.assertEqual(entry.name, "Dynamic Dataset")


class TestDatasetTable(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DatasetTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()

    def test_insert_and_conflict(self):
        entry = {
            "name": "test_dataset",
            "data_type": "image",
            "tags": ["tag1", "tag2"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [100, 100],
            "dtypes": {"data": "float32"},
        }
        self.table.insert(entry)
        conflicting = {
            "name": "conflict_dataset",
            "data_type": "image",
            "tags": ["tag2"],
            "description": "Another dataset",
            "path": "/other/path",
            "shape": [50, 50],
            "dtypes": {"data": "float32"},
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(conflicting)

    def test_insert_success(self):
        entry = {
            "name": "test_dataset2",
            "data_type": "image",
            "tags": ["tagX", "tagY"],
            "description": "Yet another dataset",
            "path": "/another/data",
            "shape": [200, 200],
            "dtypes": {"data": "float32"},
        }
        dataset_id = self.table.insert(entry)

        stored = self.table.get_by_id(dataset_id)
        self.assertEqual(stored["name"], "test_dataset2")
        self.assertEqual(stored["tags"], ["tagX", "tagY"])

    def test_update_by_id_conflict_and_success(self):
        entry = {
            "name": "dataset_to_update",
            "data_type": "image",
            "tags": ["special"],
            "description": "To update",
            "path": "/update/path",
            "shape": [100, 100],
            "dtypes": {"data": "float32"},
        }
        dataset_id = self.table.insert(entry)
        other = {
            "name": "other_dataset",
            "data_type": "image",
            "tags": ["other"],
            "description": "conflicting",
            "path": "/other",
            "shape": [20, 20],
            "dtypes": {"data": "float32"},
        }
        self.table.insert(other)
        with self.assertRaises(FedbiomedError):
            self.table.update_by_id(dataset_id, {"tags": ["other", "special"]})
        self.table.update_by_id(dataset_id, {"tags": ["unique"]})
        updated = self.table.get_by_id(dataset_id)
        self.assertIn("unique", updated["tags"])

    def test_get_by_id_methods(self):
        entry = {
            "name": "dataset_get",
            "data_type": "image",
            "tags": ["findme"],
            "description": "Get this",
            "path": "/findme",
            "shape": [25, 25],
            "dtypes": {"data": "float32"},
        }
        dataset_id = self.table.insert(entry)
        found = self.table.get_by_id(dataset_id)
        self.assertEqual(found["name"], "dataset_get")
        not_found = self.table.get_by_id("NON_EXISTENT")
        self.assertIsNone(not_found)
        validated = self.table.get_validated_entry(dataset_id)
        self.assertEqual(validated.name, "dataset_get")


class TestDynamicDatasetTable(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DynamicDatasetTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()

    def test_insert_and_get_by_id(self):
        entry = {
            "path": "/path/to/dynamic",
            "researcher_id": "res_001",
            "experiment_id": "exp_001",
            "processing_id": "proc_001",
            "parent_dataset_id": "dataset_001",
            "name": "Dynamic Dataset 1",
        }
        dataset_id = self.table.insert(entry)
        found = self.table.get_by_id(dataset_id)
        self.assertEqual(found["path"], "/path/to/dynamic")
        self.assertEqual(found["researcher_id"], "res_001")
        self.assertEqual(found["experiment_id"], "exp_001")
        self.assertEqual(found["name"], "Dynamic Dataset 1")
        not_found = self.table.get_by_id("NON_EXISTENT")
        self.assertIsNone(not_found)

    def test_collect_subtree(self):
        dataset_root_id = "dataset_root_id"
        dyn_dataset_child1 = {
            "path": "/path/to/dynamic/child1",
            "researcher_id": "res_child1",
            "experiment_id": "exp_child1",
            "processing_id": "proc_child1",
            "parent_dataset_id": dataset_root_id,
        }
        dyn_dataset_child1_id = self.table.insert(dyn_dataset_child1)
        dyn_dataset_grandchild1_id = {
            "path": "/path/to/dynamic/grandchild1",
            "researcher_id": "res_grandchild1",
            "experiment_id": "exp_grandchild1",
            "processing_id": "proc_grandchild1",
            "parent_dataset_id": dyn_dataset_child1_id,
        }
        dyn_dataset_grandchild1_id = self.table.insert(dyn_dataset_grandchild1_id)
        subtree = self.table.collect_subtree(dataset_root_id)
        self.assertEqual(
            subtree,
            [dyn_dataset_child1_id, dyn_dataset_grandchild1_id],
        )
        subtree_inexistent = self.table.collect_subtree("unknown_id")
        self.assertEqual(subtree_inexistent, [])


class TestDlpTable(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DlpTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()

    def test_insert_invalid_target_type(self):
        entry = {
            "dlp_id": "dlp_123",
            "dlp_name": "dlp_invalid",
            "target_dataset_type": "not_a_type",
            "key_paths": "/path/to/plan",
            "loading_blocks": {"block1": "block_id1"},
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(entry)

    def test_insert_short_name(self):
        entry = {
            "dlp_id": "dlp_124",
            "dlp_name": "abc",
            "target_dataset_type": DatasetTypes.IMAGES.value,
            "key_paths": "/path/to/plan",
            "loading_blocks": {"block1": "block_id1"},
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(entry)

    def test_insert_non_unique_name(self):
        entry = {
            "dlp_id": "dlp_125",
            "dlp_name": "unique_name",
            "target_dataset_type": DatasetTypes.TABULAR.value,
            "key_paths": "/plan1",
            "loading_blocks": {"block1": "block_id1"},
        }
        self.table.insert(entry)
        duplicate = {
            "dlp_id": "dlp_999",
            "dlp_name": "unique_name",
            "target_dataset_type": DatasetTypes.MEDNIST.value,
            "key_paths": "/plan2",
            "loading_blocks": {"block1": "block_id1"},
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(duplicate)

    def test_insert_success(self):
        entry = {
            "dlp_id": "dlp_100",
            "dlp_name": "valid_dlp",
            "target_dataset_type": DatasetTypes.MEDICAL_FOLDER.value,
            "key_paths": "/planX",
            "loading_blocks": {"block1": "block_id1"},
        }
        self.table.insert(entry)

        stored = self.table.get_by_id("dlp_100")
        self.assertEqual(stored["dlp_name"], "valid_dlp")
        self.assertEqual(stored["loading_blocks"], {"block1": "block_id1"})

    def test_list_by_target_dataset_type(self):
        entry = {
            "dlp_id": "dlp_101",
            "dlp_name": "dlp_listed",
            "target_dataset_type": DatasetTypes.TABULAR.value,
            "key_paths": "/planY",
            "loading_blocks": {"block1": "block_id1"},
        }
        self.table.insert(entry)
        with self.assertRaises(FedbiomedError):
            self.table.list_by_target_dataset_type("invalid")
        result = self.table.list_by_target_dataset_type(DatasetTypes.TABULAR.value)
        self.assertTrue(any(d["dlp_name"] == "dlp_listed" for d in result))


class TestNodeProcessStateTables(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.state_table = NodeProcessStateTable(self.dbfile.name)
        self.history_table = NodeProcessStateHistoryTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()

    def test_current_state_table_upserts_by_node_id(self):
        first_entry = {
            "node_id": "node-1",
            "node_name": "Node 1",
            "pid": 111,
            "state": "running",
            "action": "start",
            "reason": "process_started",
            "node_args": {
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": False,
                "debug": True,
            },
            "background": True,
        }
        second_entry = {
            "node_id": "node-1",
            "node_name": "Node 1",
            "pid": 222,
            "state": "stopped",
            "action": "stop",
            "reason": "stop_requested",
        }

        self.state_table.update_or_insert_by_id("node-1", first_entry)
        self.state_table.update_or_insert_by_id("node-1", second_entry)

        stored = self.state_table.get_by_id("node-1")
        self.assertEqual(stored["pid"], 222)
        self.assertEqual(stored["state"], "stopped")
        self.assertEqual(len(self.state_table.all()), 1)

    def test_process_state_entry_accepts_legacy_records(self):
        entry = NodeProcessStateEntry.from_dict(
            {
                "node_id": "node-1",
                "node_name": "Node 1",
                "pid": 111,
                "state": "running",
                "action": "start",
            }
        )

        self.assertIsNone(entry.node_args)
        self.assertIsNone(entry.background)

    def test_process_state_table_stores_execution_settings(self):
        entry = {
            "node_id": "node-1",
            "node_name": "Node 1",
            "pid": 111,
            "state": "running",
            "action": "start",
            "node_args": {
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": False,
                "debug": True,
            },
            "background": True,
        }

        self.state_table.update_or_insert_by_id("node-1", entry)
        stored = self.state_table.get_by_id("node-1")

        self.assertEqual(stored["node_args"], entry["node_args"])
        self.assertTrue(stored["background"])

    def test_history_table_inserts_multiple_entries_for_same_pid(self):
        first_entry = {
            "node_id": "node-1",
            "node_name": "Node 1",
            "pid": 333,
            "state": "starting",
            "action": "start",
            "reason": "start_requested",
        }
        second_entry = {
            "node_id": "node-1",
            "node_name": "Node 1",
            "pid": 333,
            "state": "running",
            "action": "start",
            "reason": "process_started",
        }

        self.history_table.insert(first_entry)
        self.history_table.insert(second_entry)

        stored = self.history_table.get_all_by_value("pid", 333)
        self.assertEqual(len(stored), 2)
        self.assertEqual([entry["state"] for entry in stored], ["starting", "running"])


class TestNodeConnectionStateTables(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.state_table = NodeConnectionStateTable(self.dbfile.name)
        self.history_table = NodeConnectionStateHistoryTable(self.dbfile.name)
        self.entry = {
            "node_id": "node-1",
            "state": "failed",
            "host": "localhost",
            "port": "50051",
            "reason": "handshake failed",
            "certificate": {"subject": "CN=RESEARCHER_1"},
            "updated_at": "2026-08-01T10:00:00Z",
        }

    def tearDown(self):
        self.dbfile.close()

    def test_current_state_table_replaces_by_node_id(self):
        self.state_table.replace_by_id("node-1", dict(self.entry))
        self.state_table.replace_by_id(
            "node-1",
            {
                "node_id": "node-1",
                "state": "connected",
                "host": "localhost",
                "port": "50051",
                "mtls": True,
                "identity_verified": False,
                "updated_at": "2026-08-20T10:00:00Z",
            },
        )

        stored = self.state_table.get_by_id("node-1")
        self.assertEqual(stored["state"], "connected")
        self.assertFalse(stored["identity_verified"])
        self.assertEqual(len(self.state_table.all()), 1)
        # Replaced, not merged: the previous state's fields are gone
        self.assertNotIn("reason", stored)
        self.assertNotIn("certificate", stored)

    def test_current_state_table_rejects_another_node_id(self):
        with self.assertRaises(FedbiomedError):
            self.state_table.replace_by_id("node-2", dict(self.entry))

    def test_connection_state_entry_keeps_certificate_fields(self):
        self.state_table.replace_by_id("node-1", dict(self.entry))

        entry = NodeConnectionStateEntry.from_dict(self.state_table.get_by_id("node-1"))
        self.assertEqual(entry.certificate, {"subject": "CN=RESEARCHER_1"})
        self.assertIsNone(entry.researcher_id)

    def test_history_table_inserts_multiple_entries_for_same_node(self):
        entry = {**self.entry, "node_id": "node-history"}
        self.history_table.insert(dict(entry))
        self.history_table.insert({**entry, "state": "connected"})

        stored = self.history_table.get_all_by_value("node_id", "node-history")
        self.assertEqual([entry["state"] for entry in stored], ["failed", "connected"])

    def test_history_table_deletes_entries_older_than_cutoff(self):
        for updated_at in (
            "2026-06-01T10:00:00Z",
            "2026-08-01T10:00:00Z",
            "2026-08-20T10:00:00Z",
        ):
            self.history_table.insert(
                {**self.entry, "node_id": "node-cleanup", "updated_at": updated_at}
            )

        self.history_table.delete_older_than("2026-07-21T10:00:00Z")

        stored = self.history_table.get_all_by_value("node_id", "node-cleanup")
        self.assertEqual(
            [entry["updated_at"] for entry in stored],
            ["2026-08-01T10:00:00Z", "2026-08-20T10:00:00Z"],
        )

    def test_history_table_does_not_remove_when_nothing_is_stale(self):
        """TinyDB rewrites the whole database file on a removal, matching or not."""
        self.history_table.insert({**self.entry, "node_id": "node-fresh"})

        with patch.object(self.history_table._table, "remove") as remove:
            removed = self.history_table.delete_older_than("2020-01-01T00:00:00Z")

        remove.assert_not_called()
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
