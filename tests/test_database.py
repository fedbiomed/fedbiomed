import tempfile
import unittest

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_dataclasses import DatasetEntry
from fedbiomed.node.dataset_manager._db_tables import DatasetTable, DlpTable


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


class TestDlpTable(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=True)
        self.table = DlpTable(self.dbfile.name)

    def tearDown(self):
        self.dbfile.close()

    def test_insert_invalid_target_type(self):
        entry = {
            "dlp_id": "dlp_123",
            "name": "dlp_invalid",
            "target_dataset_type": "not_a_type",
            "loading_plan_path": "/path/to/plan",
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(entry)

    def test_insert_short_name(self):
        entry = {
            "dlp_id": "dlp_124",
            "name": "abc",
            "target_dataset_type": DatasetTypes.IMAGES.value,
            "loading_plan_path": "/path/to/plan",
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(entry)

    def test_insert_non_unique_name(self):
        entry = {
            "dlp_id": "dlp_125",
            "name": "unique_name",
            "target_dataset_type": DatasetTypes.TABULAR.value,
            "loading_plan_path": "/plan1",
        }
        self.table.insert(entry)
        duplicate = {
            "dlp_id": "dlp_999",
            "name": "unique_name",
            "target_dataset_type": DatasetTypes.MEDNIST.value,
            "loading_plan_path": "/plan2",
        }
        with self.assertRaises(FedbiomedError):
            self.table.insert(duplicate)

    def test_insert_success(self):
        entry = {
            "dlp_id": "dlp_100",
            "name": "valid_dlp",
            "target_dataset_type": DatasetTypes.MEDICAL_FOLDER.value,
            "loading_plan_path": "/planX",
        }
        self.table.insert(entry)

    def test_list_by_target_dataset_type(self):
        entry = {
            "dlp_id": "dlp_101",
            "name": "dlp_listed",
            "target_dataset_type": DatasetTypes.TABULAR.value,
            "loading_plan_path": "/planY",
        }
        self.table.insert(entry)
        with self.assertRaises(FedbiomedError):
            self.table.list_by_target_dataset_type("invalid")
        result = self.table.list_by_target_dataset_type(DatasetTypes.TABULAR.value)
        self.assertTrue(any(d["name"] == "dlp_listed" for d in result))


if __name__ == "__main__":
    unittest.main()
