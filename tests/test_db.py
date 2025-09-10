# ---------- Test function performing CRUD on your payload ----------
import pytest

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db import DatasetDB

# @pytest.fixture
# def setup_and_teardown_db():
#     # Setup: create a fresh in-memory DB
#     memdb = TinyDB(path="test_database.json", storage=JSONStorage)
#     yield memdb
#     # Teardown: clear the DB after test
#     memdb.drop_tables()


@pytest.fixture
def database(tmp_path):
    db_path = tmp_path / "test_database.json"
    database = DatasetDB(path=str(db_path), table_name="test_datasets")
    yield database
    database._db.close()


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
                "dtypes": [],
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
                "dtypes": [
                    "int64",
                    "float64",
                    "float64",
                    "float64",
                    "int64",
                    "int64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "int64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "int64",
                    "float64",
                    "float64",
                    "float64",
                ],
                "dataset_parameters": None,
            },
        }
    }
    return datasets


def test_init_db(database):
    repo = database
    assert repo is not None
    assert isinstance(repo, DatasetDB)
    assert repo._database.name == "test_datasets"
    assert repo._database is not None
    assert repo._database.all() == []


def test_create_dataset(database, datasets):
    payload = datasets["Datasets"]

    # Create first dataset
    ds_1 = payload["1"]
    doc_id_1 = database.create(ds_1)
    assert isinstance(doc_id_1, int)
    db_all = database._list()
    assert len(db_all) == 1
    assert db_all[0]["dataset_id"] == ds_1["dataset_id"]

    # Create second dataset
    ds_2 = payload["2"]
    doc_id_2 = database.create(ds_2)
    assert isinstance(doc_id_2, int) and doc_id_2 != doc_id_1
    db_all = database._list()
    assert len(db_all) == 2
    ids = [d["dataset_id"] for d in db_all]
    assert ds_1["dataset_id"] in ids and ds_2["dataset_id"] in ids

    # Attempt to create dataset with same dataset_id should fail
    with pytest.raises(FedbiomedError):
        database.create(ds_1)

    # Attempt to create dataset with conflicting tags should fail
    ds_conflict = ds_2.copy()
    ds_conflict["dataset_id"] = "new_id"
    with pytest.raises(FedbiomedError):
        database.create(ds_conflict)

    print("Create test passed.")


def test_get_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.create(ds_1)
    database.create(ds_2)

    # Get existing dataset
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is not None and got_1["name"] == ds_1["name"]

    # Get non-existing dataset
    got_none = database.get_by_id("non_existing_id")
    assert got_none is None

    # Get by tag
    got_by_tag = database.get_by_tag(["my-data"])
    assert len(got_by_tag) == 1 and got_by_tag[0]["dataset_id"] == ds_2["dataset_id"]

    # Get by tags
    got_by_tags = database.get_by_tag(["my-data", "your-data"])
    assert len(got_by_tags) == 1 and got_by_tags[0]["dataset_id"] == ds_2["dataset_id"]

    # Get by non-existing tag
    got_none_tag = database.get_by_tag(["non-existing-tag"])
    assert len(got_none_tag) == 0

    print("Get test passed.")


def test_update_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.create(ds_1)
    database.create(ds_2)

    # Update existing dataset
    update_fields = {"description": "Updated MNIST description."}
    updated_ids = database.update_by_id(
        {"dataset_id": ds_1["dataset_id"], **update_fields}
    )
    assert len(updated_ids) == 1
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is not None and got_1["description"] == update_fields["description"]

    # Update non-existing dataset should not change anything
    updated_none = database.update_by_id(
        {"dataset_id": "non_existing_id", "description": "No change"}
    )
    assert len(updated_none) == 0

    # Attempt to update to conflicting tags should fail
    with pytest.raises(FedbiomedError):
        database.update_by_id({"dataset_id": ds_1["dataset_id"], "tags": ds_2["tags"]})

    print("Update test passed.")


def test_delete_dataset(database, datasets):
    payload = datasets["Datasets"]
    ds_1 = payload["1"]
    ds_2 = payload["2"]
    database.create(ds_1)
    database.create(ds_2)

    # Delete existing dataset
    deleted_ids = database.delete_by_id(ds_1["dataset_id"])
    assert len(deleted_ids) == 1
    got_1 = database.get_by_id(ds_1["dataset_id"])
    assert got_1 is None

    # Delete non-existing dataset should not change anything
    deleted_none = database.delete_by_id("non_existing_id")
    assert len(deleted_none) == 0

    remaining = database._database.all()
    assert len(remaining) == 1 and remaining[0]["dataset_id"] == ds_2["dataset_id"]

    print("Delete test passed.")
