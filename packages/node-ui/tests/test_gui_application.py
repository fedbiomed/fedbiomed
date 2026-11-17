"""Check GUI assets and the composed Node API."""

import shlex

import pytest
from flask_jwt_extended import create_access_token

from fedbiomed.common.constants import NODE_DATA_FOLDER
from fedbiomed.node.config import node_component


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


@pytest.mark.parametrize("asset_state", ["missing", "empty", "index_is_directory"])
def test_gui_reports_missing_frontend_at_startup(tmp_path, monkeypatch, asset_state):
    from fedbiomed_gui.server.application import create_app as create_gui

    assets = tmp_path / "build"
    if asset_state != "missing":
        assets.mkdir()
    if asset_state == "index_is_directory":
        (assets / "index.html").mkdir()
    node = tmp_path / "node with spaces"

    with pytest.raises(RuntimeError, match="GUI frontend assets are missing") as exc:
        make_app(node, monkeypatch, create_gui, build_dir=assets)

    message = str(exc.value)
    assert str(assets / "index.html") in message
    assert (
        f"fedbiomed node --path {shlex.quote(str(node))} gui start --recreate"
        in message
    )
    assert "source checkout" in message
    assert "Yarn" in message
    assert "reinstall fedbiomed-gui" in message
