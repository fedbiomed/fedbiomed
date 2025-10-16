# ---------- Test function performing CRUD on your payload ----------
import uuid

import pytest

from fedbiomed.common.db import TinyDBConnector
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_tables import DatasetTable, DlpTable


@pytest.fixture
def database(tmp_path):
    # Reset the singleton to ensure test isolation
    TinyDBConnector._instance = None

    # Create a unique database path for this test
    db_path = tmp_path / f"test_database_{uuid.uuid4().hex}.json"
    database = DatasetTable(path=str(db_path))
    yield database
    # Reset singleton after test
    TinyDBConnector._instance = None


@pytest.fixture
def datasets():
    datasets = {
        "Datasets": {
            "1": {
                "name": "MNIST",
                "data_type": "default",
                "tags": ["#MNIST", "#dataset"],
                "description": "MNIST database",
                "shape": [60000, 1, 28, 28],
                "path": "<p>/dev/data",
                "dataset_id": "dataset_6d3d3185-3635-4de6-86ff-1635a88ccd54",
                "dtypes": {},
                "dataset_parameters": None,
            },
            "2": {
                "name": "Test dataset",
                "data_type": "csv",
                "tags": ["my-data", "your-data"],
                "description": "Tes description.",
                "shape": [300, 20],
                "path": "<p>/data/adni_synth_clients/adni_client1.csv",
                "dataset_id": "dataset_85b031ea-b89f-44bb-9045-ff4396bac1d2",
                "dtypes": {
                    "col1": "int64",
                    "col2": "float64",
                    "col3": "float64",
                    "col4": "float64",
                    "col5": "int64",
                    "col6": "int64",
                    "col7": "float64",
                    "col8": "float64",
                    "col9": "float64",
                    "col10": "float64",
                    "col11": "int64",
                    "col12": "float64",
                    "col13": "float64",
                    "col14": "float64",
                    "col15": "float64",
                    "col16": "float64",
                    "col17": "int64",
                    "col18": "float64",
                    "col19": "float64",
                    "col20": "float64",
                },
                "dataset_parameters": None,
            },
        }
    }
    return datasets


@pytest.fixture
def dlp_database(tmp_path):
    # Reset the singleton to ensure test isolation
    TinyDBConnector._instance = None

    # Create a unique database path for this test
    db_path = tmp_path / f"test_dlp_database_{uuid.uuid4().hex}.json"
    db = DlpTable(path=str(db_path))
    yield db
    # Reset singleton after test
    TinyDBConnector._instance = None


@pytest.fixture
def dlps():
    # Use unique UUIDs to ensure DLP names are unique across tests
    unique_id = uuid.uuid4().hex[:8]
    return [
        {
            "dlp_id": "dlp_8c9782aa-c62f-421d-847c-89fb5e1a914d",
            "name": f"Custom DLP {unique_id}",
            "target_dataset_type": "medical-folder",
            "loading_plan_path": "/path/to/loading/plan/serialized_dlb_855af225-244b-455b-8297-d68a3ce7e1a5",
            "desc": "A test DLP for medical folder datasets",
        },
        {
            "dlp_id": "different_id",
            "name": f"Another Custom DLP {unique_id}",
            "target_dataset_type": "medical-folder",
            "loading_plan_path": "/path/to/loading/plan/custom_serialized_dlb_id",
            "desc": "Another test DLP for medical folder datasets",
        },
    ]


def test_init_db(database):
    repo = database
    assert repo is not None
    assert isinstance(repo, DatasetTable)
    assert repo._table_name == "Datasets"
    assert repo._table is not None
    assert repo.all() == []


def test_create_dataset(database, datasets):
    payload = datasets["Datasets"]

    # Create first dataset
    ds_1 = payload["1"]
    doc_id_1 = database.insert(ds_1)
    assert isinstance(doc_id_1, str)
    db_all = database.all()
    assert len(db_all) == 1
    assert db_all[0]["dataset_id"] == ds_1["dataset_id"]

    # Create second dataset
    ds_2 = payload["2"]
    doc_id_2 = database.insert(ds_2)
    assert isinstance(doc_id_2, str) and doc_id_2 != doc_id_1
    db_all = database.all()
    assert len(db_all) == 2
    ids = [d["dataset_id"] for d in db_all]
    assert ds_1["dataset_id"] in ids and ds_2["dataset_id"] in ids

    # Attempt to create dataset with same dataset_id should fail
    with pytest.raises(FedbiomedError):
        database.insert(ds_1)

    # Attempt to create dataset with conflicting tags should fail
    ds_conflict = ds_2.copy()
    ds_conflict["dataset_id"] = "new_id"
    with pytest.raises(FedbiomedError):
        database.insert(ds_conflict)

    print("Create test passed.")


def test_get_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.insert(ds_1)
    database.insert(ds_2)

    # Get existing dataset
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is not None and got_1["name"] == ds_1["name"]

    # Get non-existing dataset
    got_none = database.get_by_id("non_existing_id")
    assert got_none is None

    # Get by tag
    got_by_tag = database.search_by_tags(["my-data"])
    assert len(got_by_tag) == 1 and got_by_tag[0]["dataset_id"] == ds_2["dataset_id"]

    # Get by tags
    got_by_tags = database.search_by_tags(["my-data", "your-data"])
    assert len(got_by_tags) == 1 and got_by_tags[0]["dataset_id"] == ds_2["dataset_id"]

    # Get by non-existing tag
    got_none_tag = database.search_by_tags(["non-existing-tag"])
    assert len(got_none_tag) == 0

    print("Get test passed.")


def test_update_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.insert(ds_1)
    database.insert(ds_2)

    # Update existing dataset
    update_fields = {"description": "Updated MNIST description."}
    updated_dataset = database.update_by_id(ds_1["dataset_id"], update_fields)
    assert updated_dataset is not None
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is not None and got_1["description"] == update_fields["description"]

    # Update non-existing dataset should fail
    with pytest.raises(FedbiomedError):
        database.update_by_id("non_existing_id", {"description": "No change"})

    # Attempt to update to conflicting tags should fail
    with pytest.raises(FedbiomedError):
        database.update_by_id(ds_1["dataset_id"], {"tags": ds_2["tags"]})

    print("Update test passed.")


def test_delete_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.insert(ds_1)
    database.insert(ds_2)

    # Delete existing dataset
    deleted_ids = database.delete_by_id(ds_1["dataset_id"])
    assert len(deleted_ids) == 1
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is None

    # Delete non-existing dataset should not change anything
    deleted_none = database.delete_by_id("non_existing_id")
    assert len(deleted_none) == 0

    remaining = database.all()
    assert len(remaining) == 1 and remaining[0]["dataset_id"] == ds_2["dataset_id"]

    print("Delete test passed.")


def test_create_dlp(dlp_database, dlps):
    d1, d2 = dlps
    id1 = dlp_database.insert(d1)
    id2 = dlp_database.insert(d2)
    assert isinstance(id1, str) and isinstance(id2, str) and id1 != id2

    # Duplicate create should fail by dlp_id uniqueness
    with pytest.raises(FedbiomedError):
        dlp_database.insert(d1)

    # Missing dlp_id should fail
    with pytest.raises(FedbiomedError):
        dlp_database.insert(
            {
                "name": "Test",
                "target_dataset_type": "medical-folder",
                "loading_plan_path": "/path",
            }
        )


def test_get_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.insert(d1)
    dlp_database.insert(d2)

    got1 = dlp_database.get_by_id(d1["dlp_id"])
    got2 = dlp_database.get_by_id(d2["dlp_id"])
    assert got1 is not None and got1["dlp_id"] == d1["dlp_id"]
    assert got2 is not None and got2["dlp_id"] == d2["dlp_id"]

    assert dlp_database.get_by_id("missing") is None


def test_update_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.insert(d1)
    dlp_database.insert(d2)

    updated = dlp_database.update_by_id(d1["dlp_id"], {"desc": "Updated"})
    assert updated is not None
    got = dlp_database.get_by_id(d1["dlp_id"])
    assert got is not None and got["desc"] == "Updated"

    # Non-existing should raise an error
    with pytest.raises(FedbiomedError):
        dlp_database.update_by_id("missing", {"desc": "x"})


def test_delete_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.insert(d1)
    dlp_database.insert(d2)

    removed = dlp_database.delete_by_id(d1["dlp_id"])
    assert isinstance(removed, list) and len(removed) == 1
    assert dlp_database.get_by_id(d1["dlp_id"]) is None

    # Non-existing delete returns empty list
    assert dlp_database.delete_by_id("missing") == []

    remaining = dlp_database.all()
    assert len(remaining) == 1 and remaining[0]["dlp_id"] == d2["dlp_id"]


def test_list_by_dataset_type(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.insert(d1)
    dlp_database.insert(d2)

    # Valid type returns the matching DLP (API returns a single doc or None)
    got = dlp_database.list_by_target_dataset_type("medical-folder")
    assert isinstance(got, list)

    expected_ids = {d1["dlp_id"], d2["dlp_id"]}
    got_ids = {doc["dlp_id"] for doc in got}
    assert got_ids == expected_ids

    # Wrong input type
    with pytest.raises(FedbiomedError):
        dlp_database.list_by_target_dataset_type(123)  # not a str

    # Wrong string (not in enum values)
    with pytest.raises(FedbiomedError):
        dlp_database.list_by_target_dataset_type("not_valid_type")


def test_validation_error_handling(database):
    """Test validation error handling in BaseTable"""
    # Test missing required field that will cause validation errors
    invalid_dataset = {
        "name": "Test",
        "data_type": "invalid",
        "tags": ["test"],
        "description": "Test",
        "path": "/test",
        "shape": [100, 10],
        # Missing required 'dtypes' field
    }

    with pytest.raises(FedbiomedError):
        database.insert(invalid_dataset)


def test_dataclass_validation_exception(database):
    """Test that dataclass validation exceptions are properly caught and re-raised"""
    # Import the dataclass and mock it to raise an exception
    from unittest.mock import patch

    from fedbiomed.node.dataset_manager._db_dataclasses import DatasetEntry

    valid_data = {
        "name": "Test",
        "data_type": "default",
        "tags": ["test"],
        "description": "Test",
        "path": "/test",
        "shape": [100, 10],
        "dtypes": {},
    }

    # Mock the from_dict method to raise a ValueError
    with patch.object(
        DatasetEntry, "from_dict", side_effect=ValueError("Mock validation error")
    ):
        with pytest.raises(FedbiomedError) as exc_info:
            database.insert(valid_data)

        # Verify the error message includes our model name
        assert "DatasetEntry validation failed" in str(exc_info.value)


def test_get_validated_entry(database, datasets):
    """Test the get_validated_entry method"""
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    database.insert(ds_1)

    # Test getting existing entry as validated model
    validated_entry = database.get_validated_entry(ds_1["dataset_id"])
    assert validated_entry is not None
    assert validated_entry.name == ds_1["name"]
    assert validated_entry.dataset_id == ds_1["dataset_id"]

    # Test getting non-existing entry
    validated_none = database.get_validated_entry("non_existing_id")
    assert validated_none is None


def test_update_id_prevention(database, datasets):
    """Test that updating the ID field is prevented"""
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    database.insert(ds_1)

    # Attempt to change the dataset_id should fail
    with pytest.raises(FedbiomedError):
        database.update_by_id(ds_1["dataset_id"], {"dataset_id": "new_id"})


def test_dlp_invalid_dataset_type(dlp_database):
    """Test DLP insertion with invalid dataset type"""
    invalid_dlp = {
        "dlp_id": "test_id",
        "name": "Test DLP",
        "target_dataset_type": "invalid_type",  # Not in DatasetTypes enum
        "loading_plan_path": "/path/to/plan",
    }

    with pytest.raises(FedbiomedError):
        dlp_database.insert(invalid_dlp)


def test_dlp_short_name_validation(dlp_database):
    """Test DLP insertion with name too short"""
    short_name_dlp = {
        "dlp_id": "test_id",
        "name": "XY",  # Less than 4 characters
        "target_dataset_type": "medical-folder",
        "loading_plan_path": "/path/to/plan",
    }

    with pytest.raises(FedbiomedError):
        dlp_database.insert(short_name_dlp)
