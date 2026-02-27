import tempfile
import unittest

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import (
    DatasetEntry,
    DynamicDatasetEntry,
)
from fedbiomed.node.dataset_manager._db_tables import (
    DatasetTable,
    DlpTable,
    DynamicDatasetTable,
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


if __name__ == "__main__":
    unittest.main()
