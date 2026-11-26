"""Check GUI assets and the composed Node API."""

import pytest
from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.node.config import node_component
from flask_jwt_extended import create_access_token


def make_app(root, monkeypatch, factory, **kwargs):
    node_component.initiate(str(root))
    monkeypatch.setenv("DATA_PATH", str(root / NODE_DATA_FOLDER))
    return factory(root, {"TESTING": True}, **kwargs)


def token(app):
    with app.app_context():
        return {"Authorization": "Bearer " + create_access_token(identity="test")}


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


@pytest.mark.parametrize(
    "bundled, level", [(True, 1), (False, 3)], ids=["package", "checkout"]
)
def test_frontend_assets_prefer_package_over_checkout(monkeypatch, bundled, level):
    from fedbiomed_gui.server import application

    monkeypatch.setattr(application.Path, "is_dir", lambda path: bundled)
    module = application.Path(application.__file__).resolve()
    assert application.frontend_assets() == module.parents[level] / "ui/build"
