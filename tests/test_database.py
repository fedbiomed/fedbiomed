import tempfile
import unittest
from unittest.mock import patch

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.db import TinyDBConnector
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import DatasetEntry
from fedbiomed.node.dataset_manager._db_tables import DatasetTable, DlpTable


def _reset_tinydb_singleton() -> None:
    """Reset TinyDBConnector singleton to ensure test isolation.

    TinyDBConnector is implemented as a process-wide singleton; without resetting,
    tests that use different temp DB files can end up sharing the same TinyDB
    instance, causing cross-test contamination.
    """
    inst = getattr(TinyDBConnector, "_instance", None)
    if inst is not None:
        db = getattr(inst, "_db", None)
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    TinyDBConnector._instance = None


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


class TestDatasetTable(unittest.TestCase):
    def setUp(self):
        _reset_tinydb_singleton()
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DatasetTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()
        _reset_tinydb_singleton()

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
        self.table.insert(entry)

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


class TestDlpTable(unittest.TestCase):
    def setUp(self):
        _reset_tinydb_singleton()
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DlpTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()
        _reset_tinydb_singleton()

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


class TestSecurityLoggingDatasetTable(unittest.TestCase):
    def setUp(self):
        _reset_tinydb_singleton()
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DatasetTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()
        _reset_tinydb_singleton()

    def _assert_security_event_called(
        self, mock_security_event, *, operation: str, status: str, **expected_fields
    ):
        calls = [c.kwargs for c in mock_security_event.call_args_list]
        for kwargs in calls:
            if kwargs.get("operation") != operation:
                continue
            if kwargs.get("status") != status:
                continue
            if all(kwargs.get(k) == v for k, v in expected_fields.items()):
                return
        self.fail(
            f"Expected security_event(operation={operation!r}, status={status!r}, fields={expected_fields!r}) not found. "
            f"Calls were: {calls!r}"
        )

    def test_security_log_insert_includes_dataset_id(self):
        entry = {
            "name": "sec_ds",
            "data_type": "image",
            "tags": ["tag1"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            dataset_id = self.table.insert(entry)
            self._assert_security_event_called(
                sec,
                operation="dataset_create",
                status="success",
                dataset_id=dataset_id,
                entry_name="sec_ds",
            )

    def test_security_log_insert_includes_dlp_id_when_present(self):
        entry = {
            "name": "sec_ds_dlp",
            "data_type": "image",
            "tags": ["tag1"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
            "dlp_id": "dlp_123",
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            dataset_id = self.table.insert(entry)
            self._assert_security_event_called(
                sec,
                operation="dataset_create",
                status="success",
                dataset_id=dataset_id,
                dlp_id="dlp_123",
                entry_name="sec_ds_dlp",
            )

    def test_security_log_all_emits_dataset_list(self):
        entry = {
            "name": "sec_list",
            "data_type": "image",
            "tags": ["t"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            self.table.insert(entry)
            sec.reset_mock()
            rows = self.table.all()
            self.assertEqual(len(rows), 1)
            self._assert_security_event_called(
                sec,
                operation="dataset_list",
                status="success",
                record_count=1,
            )

    def test_security_log_get_by_id_emits_dataset_read(self):
        entry = {
            "name": "sec_read",
            "data_type": "image",
            "tags": ["t"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            dataset_id = self.table.insert(entry)
            sec.reset_mock()
            _ = self.table.get_by_id(dataset_id)
            self._assert_security_event_called(
                sec,
                operation="dataset_read",
                status="success",
                dataset_id=dataset_id,
                entry_name="sec_read",
            )

    def test_security_log_get_by_id_not_found_emits_not_found(self):
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            _ = self.table.get_by_id("NON_EXISTENT")
            self._assert_security_event_called(
                sec,
                operation="dataset_read",
                status="not_found",
                dataset_id="NON_EXISTENT",
            )

    def test_security_log_update_by_id_emits_dataset_update(self):
        entry = {
            "name": "sec_update",
            "data_type": "image",
            "tags": ["t"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            dataset_id = self.table.insert(entry)
            sec.reset_mock()
            self.table.update_by_id(dataset_id, {"description": "changed"})
            self._assert_security_event_called(
                sec,
                operation="dataset_update",
                status="success",
                dataset_id=dataset_id,
                entry_name="sec_update",
            )

    def test_security_log_delete_by_id_emits_dataset_delete(self):
        entry = {
            "name": "sec_delete",
            "data_type": "image",
            "tags": ["t"],
            "description": "A test dataset",
            "path": "/path/to/data",
            "shape": [10, 10],
            "dtypes": {"data": "float32"},
        }
        with patch(
            "fedbiomed.node.dataset_manager._db_tables.logger.security_event"
        ) as sec:
            dataset_id = self.table.insert(entry)
            sec.reset_mock()
            self.table.delete_by_id(dataset_id)
            self._assert_security_event_called(
                sec,
                operation="dataset_delete",
                status="success",
                dataset_id=dataset_id,
                entry_name="sec_delete",
                data_type="image",
            )


if __name__ == "__main__":
    unittest.main()
