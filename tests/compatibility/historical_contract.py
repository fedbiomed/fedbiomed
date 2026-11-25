"""Run with installed packages: seed old node files or check API contracts.

The driver runs seed with released packages, then check with candidate packages
against the same node directory. No hand-written imitation of an old DB is used.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def create_app(root):
    from fedbiomed.common.constants import NODE_DATA_FOLDER
    from fedbiomed_node_api.application import create_app as factory

    os.environ["DATA_PATH"] = str(root / NODE_DATA_FOLDER)
    return factory(str(root), {"TESTING": True})


def seed(root):
    from fedbiomed.node.config import node_component
    from fedbiomed.node.dataset_manager import DatasetManager
    from fedbiomed_node_api.utils import set_password_hash

    config = node_component.initiate(str(root))
    config.set("default", "name", "Historical custom node")
    config.set("security", "training_plan_approval", "True")
    config.write()
    app = create_app(root)
    # A non-default password catches accidental recreation of the user DB.
    app.extensions["node_api_services"]["user_database"].table("Users").update(
        {"password_hash": set_password_hash("historical-test-password")}
    )
    csv = Path(app.config["DATA_PATH_RW"]) / "historical.csv"
    csv.write_text("feature,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    dataset_id = DatasetManager(app.config["NODE_DB_PATH"]).add_database(
        name="Historical sample",
        data_type="csv",
        tags=["historical"],
        description="Historical compatibility fixture",
        path=str(csv),
    )
    (root / "expected.json").write_text(
        json.dumps({"node_id": app.config["ID"], "dataset_id": dataset_id}),
        encoding="utf-8",
    )


def check(root):
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    # Do not initiate a new node: reopen the existing config and databases.
    app = create_app(root)
    client = app.test_client()
    assert client.get("/api/config/node-id").status_code == 401

    # Login also exercises the user database persisted by the seed version.
    login = client.post(
        "/api/auth/token/login",
        json={"email": "admin@fedbiomed.gui", "password": "historical-test-password"},
    )
    assert login.status_code == 200, login.get_data(as_text=True)

    def result(response):
        assert response.status_code == 200, response.get_data(as_text=True)
        assert response.is_json
        body = response.get_json()
        assert {"success", "result", "message"} <= body.keys()
        assert body["success"] is True
        return body["result"]

    tokens = result(login)
    assert isinstance(tokens["access_token"], str) and tokens["access_token"]
    assert isinstance(tokens["refresh_token"], str) and tokens["refresh_token"]
    headers = {"Authorization": "Bearer " + tokens["access_token"]}

    assert result(client.get("/api/config/node-id", headers=headers)) == {
        "node_id": expected["node_id"]
    }
    datasets = result(client.post("/api/datasets/list", json={}, headers=headers))
    assert isinstance(datasets, list) and len(datasets) == 1
    assert datasets[0]["dataset_id"] == expected["dataset_id"]
    assert datasets[0]["name"] == "Historical sample"
    assert datasets[0]["tags"] == ["historical"]
    preview = result(
        client.post(
            "/api/datasets/preview",
            json={"dataset_id": expected["dataset_id"]},
            headers=headers,
        )
    )
    assert preview["data_preview"]["columns"] == ["feature", "target"]
    assert preview["data_preview"]["data"] == [[1, 0], [2, 1], [3, 0]]
    settings = result(client.get("/api/config/node-environ", headers=headers))
    assert settings["name"] == "Historical custom node"
    assert settings["training_plan_approval"].lower() == "true"
    check_dataset_writes(client, headers, result, root, expected["dataset_id"])
    print("API contracts and persisted node configuration/user/dataset data passed.")


def check_dataset_writes(client, headers, result, root, historical_id):
    """Exercise CLI add/list/delete and API update (no CLI update exists)."""
    from fedbiomed.common.constants import NODE_DATA_FOLDER

    (root / NODE_DATA_FOLDER / "write-test.csv").write_text(
        "feature,target\n4,0\n5,1\n6,0\n", encoding="utf-8"
    )

    def post(route, body):
        return result(client.post("/api/datasets/" + route, json=body, headers=headers))

    description = root / "write-dataset.json"
    description.write_text(
        json.dumps(
            {
                "name": "Write sample",
                "data_type": "csv",
                "tags": "write-test",
                "description": "Created through the CLI",
                "path": str(root / NODE_DATA_FOLDER / "write-test.csv"),
            }
        ),
        encoding="utf-8",
    )

    def cli(*args, input_text=""):
        command = [
            sys.executable,
            "-c",
            "from fedbiomed.cli import run; run()",
            "node",
            "--path",
            str(root),
            "dataset",
            *args,
        ]
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    cli("add", "--file", str(description))
    listing = cli("list")
    assert "Write sample" in listing and historical_id in listing, listing
    created = next(item for item in post("list", {}) if item["name"] == "Write sample")
    dataset_id = created["dataset_id"]
    assert dataset_id != historical_id
    assert created["name"] == "Write sample"
    assert created["tags"] == ["write-test"]
    updated = post(
        "update",
        {
            "dataset_id": dataset_id,
            "name": "Updated sample",
            "tags": ["updated-test"],
            "desc": "Updated through the API",
        },
    )
    assert updated["dataset_id"] == dataset_id
    records = {item["dataset_id"]: item for item in post("list", {})}
    assert set(records) == {historical_id, dataset_id}
    assert records[dataset_id]["name"] == "Updated sample"
    assert records[dataset_id]["tags"] == ["updated-test"]
    assert records[dataset_id]["description"] == "Updated through the API"
    listing = cli("list")
    assert "Updated sample" in listing and dataset_id in listing, listing
    # The fixture inserts the historical dataset first and the CLI dataset second.
    # Assert the displayed menu before accepting this as a targeted deletion.
    deletion = cli("delete", input_text="2\n")
    assert "1) Historical sample" in deletion and "2) Updated sample" in deletion, (
        deletion
    )
    listing = cli("list")
    assert dataset_id not in listing and historical_id in listing, listing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["seed", "check", "fresh"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.mode in ("seed", "fresh"):
        seed(args.root)
    if args.mode in ("check", "fresh"):
        check(args.root)
