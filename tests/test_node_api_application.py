"""Integration checks for API and GUI application composition."""

from flask_jwt_extended import create_access_token

from fedbiomed.node.config import node_component
from fedbiomed_node_api.application import create_app


def make_app(root, monkeypatch, factory=create_app, **kwargs):
    node_component.initiate(str(root))
    monkeypatch.setenv("DATA_PATH", str(root / "data"))
    return factory(root, {"TESTING": True}, **kwargs)


def token(app):
    with app.app_context():
        return {"Authorization": "Bearer " + create_access_token(identity="test")}


def test_api_has_no_frontend_and_requires_authentication(tmp_path, monkeypatch):
    app = make_app(tmp_path / "node", monkeypatch)
    client = app.test_client()
    assert app.static_folder is None
    assert client.get("/").status_code == 404
    assert client.get("/static/missing.js").status_code == 404
    assert client.get("/api/config/node-id").status_code == 401
    response = client.get("/api/config/node-id", headers=token(app))
    assert response.status_code == 200
    assert response.json["result"]["node_id"] == app.config["ID"]
    assert "DEFAULT_ADMIN_CREDENTIAL" not in app.config


def test_two_apps_keep_node_state_separate(tmp_path, monkeypatch):
    first = make_app(tmp_path / "first", monkeypatch)
    second = make_app(tmp_path / "second", monkeypatch)
    assert first.config["ID"] != second.config["ID"]
    for app in (first, second, first):
        result = app.test_client().get("/api/config/node-id", headers=token(app))
        assert result.json["result"]["node_id"] == app.config["ID"]
    services = first.extensions["node_api_services"]
    other = second.extensions["node_api_services"]
    services["user_database"].table("Users").insert({"user_email": "first-only"})
    assert len(services["user_database"].table("Users")) == 2
    assert len(other["user_database"].table("Users")) == 1
    services["repository_file_sizes"]["test"] = 12
    assert not other["repository_file_sizes"]


def test_gui_serves_assets_and_api(tmp_path, monkeypatch):
    from fedbiomed_gui.server.application import create_app as create_gui

    assets = tmp_path / "build"
    assets.mkdir()
    (assets / "index.html").write_text("<html>GUI</html>")
    (assets / "main.js").write_text("window.test = true;")
    app = make_app(tmp_path / "node", monkeypatch, create_gui, build_dir=assets)
    client = app.test_client()
    assert client.get("/").data == b"<html>GUI</html>"
    assert client.get("/datasets").data == b"<html>GUI</html>"
    assert client.get("/main.js").data == b"window.test = true;"
    assert client.get("/build/main.js").status_code == 200
    assert client.get("/api/config/node-id", headers=token(app)).status_code == 200


def test_gui_rejects_missing_assets_before_node_initialization(tmp_path):
    import pytest

    from fedbiomed.common.exceptions import FedbiomedError
    from fedbiomed_gui.server.application import create_app as create_gui

    root = tmp_path / "missing-node"
    with pytest.raises(FedbiomedError, match="frontend assets are missing"):
        create_gui(root, build_dir=tmp_path / "missing-build")
    assert not root.exists()


def test_login_uses_application_database(tmp_path, monkeypatch):
    app = make_app(tmp_path / "node", monkeypatch)
    client = app.test_client()
    response = client.post(
        "/api/auth/token/login",
        json={
            "email": "admin@fedbiomed.gui",
            "password": "admin",
        },
    )
    assert response.status_code == 200
    assert (
        client.post(
            "/api/auth/token/login",
            json={
                "email": "admin@fedbiomed.gui",
                "password": "wrong",
            },
        ).status_code
        == 401
    )


def test_import_does_not_initialize_node(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from fedbiomed_node_api.application import create_app; "
                "import fedbiomed_node_api.routes; "
                "assert not any(n.startswith('fedbiomed_gui') for n in sys.modules)"
            ),
        ],
        env={**os.environ, "FBM_NODE_COMPONENT_ROOT": str(tmp_path / "missing")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "missing").exists()


def test_api_lists_datasets_registered_by_core(tmp_path, monkeypatch):
    from fedbiomed.node.dataset_manager import DatasetManager

    app = make_app(tmp_path / "node", monkeypatch)
    csv = tmp_path / "node" / "data" / "sample.csv"
    csv.write_text("feature,target\n1,0\n2,1\n3,0\n")
    manager = DatasetManager(app.config["NODE_DB_PATH"])
    dataset_id = manager.add_database(
        name="Sample",
        data_type="csv",
        tags=["sample"],
        description="API integration test",
        path=str(csv),
    )
    response = app.test_client().post(
        "/api/datasets/list",
        json={},
        headers=token(app),
    )
    assert response.status_code == 200
    assert [dataset["dataset_id"] for dataset in response.json["result"]] == [
        dataset_id
    ]
