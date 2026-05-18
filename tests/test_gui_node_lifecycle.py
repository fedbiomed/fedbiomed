import importlib
import sys

import pytest
from flask_jwt_extended import create_access_token

from fedbiomed.common.constants import UserRoleType
from fedbiomed.common.utils import create_fedbiomed_setup_folders


@pytest.fixture()
def gui_app(tmp_path, monkeypatch):
    root = tmp_path / "node"
    data = root / "data"
    root.mkdir()
    create_fedbiomed_setup_folders(str(root))
    data.mkdir()

    monkeypatch.setenv("FBM_NODE_COMPONENT_ROOT", str(root))
    monkeypatch.setenv("DATA_PATH", str(data))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")

    for module_name in list(sys.modules):
        if module_name.startswith("fedbiomed_gui.server"):
            sys.modules.pop(module_name)

    application = importlib.import_module("fedbiomed_gui.server.application")
    lifecycle = importlib.import_module("fedbiomed_gui.server.routes.node_lifecycle")
    db = importlib.import_module("fedbiomed_gui.server.db")

    application.app.config["TESTING"] = True
    return application.app, lifecycle, db


def _insert_user(db, *, user_id: str, role: UserRoleType):
    db.user_database.table("Users").insert(
        {
            "user_id": user_id,
            "user_email": f"{user_id}@example.org",
            "user_name": "Test",
            "user_surname": "User",
            "password_hash": "not-used",
            "user_role": role,
            "creation_date": "2026-05-18T00:00:00",
        }
    )


def _auth_header(app, *, user_id: str, role: UserRoleType):
    with app.app_context():
        token = create_access_token(
            identity=user_id,
            additional_claims={
                "email": f"{user_id}@example.org",
                "role": role,
                "name": "Test",
                "surname": "User",
            },
        )
    return {"Authorization": f"Bearer {token}"}


def _status(state="stopped"):
    return {
        "node_id": "node-1",
        "node_name": "Node 1",
        "state": state,
        "pid": None,
        "action": "status",
        "reason": None,
        "updated_at": None,
        "started_at": None,
        "stopped_at": None,
        "exit_code": None,
        "managed_by_current_process": False,
    }


def test_gui_node_lifecycle_01_status_requires_admin(gui_app, mocker):
    app, lifecycle, db = gui_app
    _insert_user(db, user_id="user-1", role=UserRoleType.USER)
    manager = mocker.MagicMock()
    manager.get_process_state.return_value = _status()
    lifecycle.node_process_manager = manager

    response = app.test_client().get(
        "/api/node/lifecycle/status",
        headers=_auth_header(app, user_id="user-1", role=UserRoleType.USER),
    )

    assert response.status_code == 403
    manager.get_process_state.assert_not_called()


def test_gui_node_lifecycle_02_status_returns_manager_state(gui_app, mocker):
    app, lifecycle, db = gui_app
    _insert_user(db, user_id="admin-1", role=UserRoleType.ADMIN)
    manager = mocker.MagicMock()
    manager.get_process_state.return_value = _status(state="running")
    lifecycle.node_process_manager = manager

    response = app.test_client().get(
        "/api/node/lifecycle/status",
        headers=_auth_header(app, user_id="admin-1", role=UserRoleType.ADMIN),
    )

    assert response.status_code == 200
    assert response.json["result"]["state"] == "running"
    manager.get_process_state.assert_called_once_with()


def test_gui_node_lifecycle_03_start_forwards_node_args_and_actor(gui_app, mocker):
    app, lifecycle, db = gui_app
    _insert_user(db, user_id="admin-1", role=UserRoleType.ADMIN)
    manager = mocker.MagicMock()
    lifecycle.node_process_manager = manager
    manager.get_process_state.return_value = _status(state="running")

    response = app.test_client().post(
        "/api/node/lifecycle/start",
        json={"gpu": False, "gpu_num": 2, "gpu_only": True, "debug": True},
        headers=_auth_header(app, user_id="admin-1", role=UserRoleType.ADMIN),
    )

    assert response.status_code == 200
    manager.start.assert_called_once_with(
        {"gpu": True, "gpu_num": 2, "gpu_only": True, "debug": True},
        actor={
            "source": "gui",
            "user_id": "admin-1",
            "email": "admin-1@example.org",
            "role": UserRoleType.ADMIN,
            "name": "Test",
            "surname": "User",
        },
    )


def test_gui_node_lifecycle_04_stop_and_restart_call_manager(gui_app, mocker):
    app, lifecycle, db = gui_app
    _insert_user(db, user_id="admin-1", role=UserRoleType.ADMIN)
    manager = mocker.MagicMock()
    lifecycle.node_process_manager = manager
    manager.get_process_state.return_value = _status(state="stopped")
    headers = _auth_header(app, user_id="admin-1", role=UserRoleType.ADMIN)

    stop_response = app.test_client().post(
        "/api/node/lifecycle/stop",
        headers=headers,
    )
    restart_response = app.test_client().post(
        "/api/node/lifecycle/restart",
        json={"gpu": True, "gpu_num": 3, "gpu_only": False, "debug": False},
        headers=headers,
    )

    assert stop_response.status_code == 200
    assert restart_response.status_code == 200
    manager.stop.assert_called_once()
    manager.restart.assert_called_once_with(
        {"gpu": True, "gpu_num": 3, "gpu_only": False, "debug": False},
        actor=manager.stop.call_args.kwargs["actor"],
    )
