# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Exercise GUI password-change restrictions with real routes and isolated storage."""

import importlib
import sys
import types
from pathlib import Path

import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, decode_token
from tinydb import Query, TinyDB
from tinydb.storages import MemoryStorage

from fedbiomed.common.constants import UserRoleType


@pytest.fixture
def gui(monkeypatch):
    # Load just the auth routes, avoiding node configuration and real databases.
    prefix = "fedbiomed_gui.server"
    for name in list(sys.modules):
        if name.startswith(prefix + "."):
            monkeypatch.delitem(sys.modules, name)
    routes = types.ModuleType(prefix + ".routes")
    routes.__path__ = [str(Path(__file__).parents[1] / "fedbiomed_gui/server/routes")]
    monkeypatch.setitem(sys.modules, routes.__name__, routes)
    database = TinyDB(storage=MemoryStorage)
    db = types.ModuleType(prefix + ".db")
    db.user_database = types.SimpleNamespace(table=database.table, query=Query)
    monkeypatch.setitem(sys.modules, db.__name__, db)
    auth = importlib.import_module(prefix + ".routes.authentication")
    importlib.import_module(prefix + ".routes.users")
    app = Flask(__name__)
    app.config.update(TESTING=True, DEBUG=False, JWT_SECRET_KEY="test-secret-" * 4)
    JWTManager(app)

    @auth.api.route("/protected")
    def protected():
        return {"ok": True}

    app.register_blueprint(auth.api)
    app.register_blueprint(auth.auth)
    user = dict(
        user_id="test-user",
        user_email="user@example.org",
        password_hash=auth.set_password_hash("InitialPass1"),
        user_role=UserRoleType.USER,
        must_change_password=True,
    )
    database.table("Users").insert(user)
    yield app, app.test_client(), database.table("Users"), auth
    database.close()
    for name in list(sys.modules):
        if name.startswith(prefix + ".") and name not in {db.__name__, routes.__name__}:
            sys.modules.pop(name, None)


def login(client, password="InitialPass1"):
    response = client.post(
        "/api/auth/token/login",
        json={"email": "user@example.org", "password": password},
    )
    assert response.status_code == 200
    tokens = response.json["result"]
    return {"Authorization": "Bearer " + tokens["access_token"]}, tokens


def test_first_login_requires_change_until_valid_new_password(gui):
    app, client, table, auth = gui
    headers, tokens = login(client)
    assert client.get("/api/protected", headers=headers).status_code == 403
    assert client.get("/api/token/auth", headers=headers).json["result"][
        "must_change_password"
    ]
    # Logging in again or refreshing cannot remove the restriction.
    login(client)
    refreshed = client.get(
        "/api/auth/token/refresh",
        headers={"Authorization": "Bearer " + tokens["refresh_token"]},
    )
    with app.app_context():
        assert decode_token(refreshed.json["result"]["access_token"])[
            "must_change_password"
        ]
    data = dict(
        email="user@example.org", old_password="InitialPass1", password="InitialPass1"
    )
    for changes in (
        {},
        {"password": "weak"},
        {"old_password": "wrong", "password": "ChangedPass2"},
        {"old_password": None},
    ):
        assert (
            client.post(
                "/api/update-password", headers=headers, json={**data, **changes}
            ).status_code
            == 400
        )
        assert client.get("/api/protected", headers=headers).status_code == 403
    assert (
        client.post(
            "/api/update-password",
            headers=headers,
            json={**data, "password": "ChangedPass2"},
        ).status_code
        == 200
    )
    assert table.all()[0]["must_change_password"] is False
    new_headers, _ = login(client, "ChangedPass2")
    assert client.get("/api/protected", headers=new_headers).status_code == 200
    assert (
        client.post(
            "/api/auth/token/login",
            json={"email": "user@example.org", "password": "InitialPass1"},
        ).status_code
        == 401
    )


@pytest.mark.parametrize(
    "previous_login, expected", [(None, True), ("2026-01-01", False)]
)
def test_legacy_accounts(gui, previous_login, expected):
    app, client, table, auth = gui
    table.update(lambda user: user.pop("must_change_password"))
    if previous_login:
        table.update({"last_login": previous_login})
    headers, _ = login(client)
    assert table.all()[0]["must_change_password"] is expected
    assert client.get("/api/protected", headers=headers).status_code == (
        403 if expected else 200
    )


def test_debug_bypass_does_not_clear_requirement(gui):
    app, client, table, auth = gui
    app.config["DEBUG"] = True
    headers, _ = login(client)
    assert client.get("/api/protected", headers=headers).status_code == 200
    assert table.all()[0]["must_change_password"] is True
    app.config["DEBUG"] = False
    assert client.get("/api/protected", headers=headers).status_code == 403


def test_reset_restricts_existing_session(gui):
    app, client, table, auth = gui
    table.update({"must_change_password": False})
    headers, _ = login(client)
    table.insert(
        dict(
            user_id="admin",
            user_email="admin@example.org",
            user_role=UserRoleType.ADMIN,
            password_hash=auth.set_password_hash("AdminPass1"),
            must_change_password=False,
        )
    )
    response = client.post(
        "/api/auth/token/login",
        json={"email": "admin@example.org", "password": "AdminPass1"},
    )
    admin_headers = {
        "Authorization": "Bearer " + response.json["result"]["access_token"]
    }
    response = client.patch(
        "/api/admin/users/reset-password",
        headers=admin_headers,
        json={"user_id": "test-user"},
    )
    assert response.status_code == 200
    assert client.get("/api/protected", headers=headers).status_code == 403
    reset_headers, _ = login(client, response.json["result"]["password"])
    assert client.get("/api/protected", headers=reset_headers).status_code == 403


def test_new_account_creation_paths_and_registration_validation(gui):
    app, client, table, auth = gui
    table.update({"must_change_password": False, "user_role": UserRoleType.ADMIN})
    headers, _ = login(client)
    data = dict(
        email="new@example.org",
        name="New",
        surname="User",
        password="NewPassword1",
        confirm="NewPassword1",
    )
    assert (
        client.post(
            "/api/auth/register", json={**data, "password": "weak", "confirm": "weak"}
        ).status_code
        == 400
    )
    response = client.post("/api/auth/register", json=data)
    assert response.status_code == 201
    request_id = response.json["result"]["request_id"]
    assert (
        client.post(
            "/api/admin/requests/approve",
            headers=headers,
            json={"request_id": request_id},
        ).status_code
        == 201
    )
    assert (
        table.get(Query().user_email == data["email"])["must_change_password"] is True
    )
    data["email"] = "created@example.org"
    assert (
        client.post("/api/admin/users/create", headers=headers, json=data).status_code
        == 201
    )
    assert (
        table.get(Query().user_email == data["email"])["must_change_password"] is True
    )
