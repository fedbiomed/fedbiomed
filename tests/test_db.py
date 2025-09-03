# ---------- Test function performing CRUD on your payload ----------
from tinydb import JSONStorage, TinyDB
from fedbiomed.node.dataset_manager._db import DatasetDB


def test_dataset_crud_on_payload():
    # In-memory DB for a clean test run
    memdb = TinyDB(path="test_database.json", storage=JSONStorage)
    repo = DatasetDB(storage=memdb.storage, name="test_datasets")

    # --- Given payload (your example) ---
    payload = {
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
                    "int64", "float64", "float64", "float64", "int64",
                    "int64", "float64", "float64", "float64", "float64",
                    "int64", "float64", "float64", "float64", "float64",
                    "float64", "int64", "float64", "float64", "float64",
                ],
                "dataset_parameters": None,
            },
        }
    }

    # ---- C: create (via bulk upsert) ----
    repo.upsert_from_payload(payload)
    all_after_create = repo.list_datasets()
    assert len(all_after_create) == 2

    # ---- R: read one ----
    ds_id_b = "dataset_85b031ea-b89f-44bb-9045-ff4396bac1d2"
    got_b = repo.read_dataset(ds_id_b)
    assert got_b and got_b["name"] == "Test dataset"

    # ---- U: update description field ----
    assert repo.update_dataset(ds_id_b, {"description": "Updated description."}) is True
    got_b2 = repo.read_dataset(ds_id_b)
    assert got_b2 and got_b2["description"] == "Updated description."

    # ---- D: delete the other dataset ----
    ds_id_a = "dataset_6d3d3185-3635-4de6-86ff-1635a88ccd54"
    assert repo.delete_dataset(ds_id_a) is True
    assert repo.read_dataset(ds_id_a) is None

    # Final list should contain only one
    final_list = repo.list_datasets()
    assert len(final_list) == 1 and final_list[0]["dataset_id"] == ds_id_b

    print("CRUD test passed. Final dataset:", final_list[0]["dataset_id"])