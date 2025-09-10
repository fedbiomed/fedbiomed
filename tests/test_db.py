# ---------- Test function performing CRUD on your payload ----------
import pytest

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db import DatasetDB, DlpDB

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


@pytest.fixture
def dlp_database(tmp_path):
    db_path = tmp_path / "test_dlp_database.json"
    db = DlpDB(path=str(db_path), table_name="test_dlp")
    yield db
    db._db.close()


@pytest.fixture
def dlps():
    return [
        {
            "dlp_id": "dlp_1",
            "name": "Plan A",
            "description": "First DLP",
            "dataset_type": "csv",
        },
        {
            "dlp_id": "dlp_2",
            "name": "Plan B",
            "description": "Second DLP",
            "dataset_type": "medical-folder",
        },
    ]


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


def test_create_dlp(dlp_database, dlps):
    d1, d2 = dlps
    id1 = dlp_database.create(d1)
    id2 = dlp_database.create(d2)
    assert isinstance(id1, int) and isinstance(id2, int) and id1 != id2

    # Duplicate create should fail by dlp_id uniqueness
    with pytest.raises(FedbiomedError):
        dlp_database.create(d1)

    # Missing dlp_id should fail
    with pytest.raises(FedbiomedError):
        dlp_database.create({"name": "No ID"})


def test_get_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.create(d1)
    dlp_database.create(d2)

    got1 = dlp_database.get_by_id("dlp_1")
    got2 = dlp_database.get_by_id("dlp_2")
    assert got1 is not None and got1["dlp_id"] == "dlp_1"
    assert got2 is not None and got2["dlp_id"] == "dlp_2"

    assert dlp_database.get_by_id("missing") is None


def test_update_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.create(d1)
    dlp_database.create(d2)

    updated = dlp_database.update_by_id({"dlp_id": "dlp_1", "description": "Updated"})
    assert isinstance(updated, list) and len(updated) == 1
    got = dlp_database.get_by_id("dlp_1")
    assert got is not None and got["description"] == "Updated"

    # Non-existing should be a no-op (empty list)
    updated_none = dlp_database.update_by_id({"dlp_id": "missing", "description": "x"})
    assert updated_none == []


def test_delete_by_id_dlp(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.create(d1)
    dlp_database.create(d2)

    removed = dlp_database.delete_by_id("dlp_1")
    assert isinstance(removed, list) and len(removed) == 1
    assert dlp_database.get_by_id("dlp_1") is None

    # Non-existing delete returns empty list
    assert dlp_database.delete_by_id("missing") == []

    remaining = dlp_database._database.all()
    assert len(remaining) == 1 and remaining[0]["dlp_id"] == "dlp_2"


def test_list_by_dataset_type(dlp_database, dlps):
    d1, d2 = dlps
    dlp_database.create(d1)
    dlp_database.create(d2)

    # Valid type returns the matching DLP (API returns a single doc or None)
    got = dlp_database.list_by_dataset_type("medical-folder")
    assert got is not None and got["dataset_type"] == "medical-folder"

    # Wrong input type
    with pytest.raises(FedbiomedError):
        dlp_database.list_by_dataset_type(123)  # not a str

    # Wrong string (not in enum values)
    with pytest.raises(FedbiomedError):
        dlp_database.list_by_dataset_type("not_valid_type")
