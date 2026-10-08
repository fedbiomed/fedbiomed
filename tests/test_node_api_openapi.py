"""Keep the public contract, installed resource and website aligned with the API."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from jsonschema import Draft202012Validator

from fedbiomed.node.config import node_component
from fedbiomed_node_api import schemas
from fedbiomed_node_api.application import create_app

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = json.loads((ROOT / "fedbiomed_node_api/openapi.json").read_text())


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = tmp_path / "node"
    node_component.initiate(str(root))
    monkeypatch.setenv("DATA_PATH", str(root / "data"))
    return create_app(root, {"TESTING": True})


def validator(schema):
    # OpenAPI 3.1 schemas use JSON Schema 2020-12, including type arrays and allOf.
    return Draft202012Validator({"components": DOCUMENT["components"], **schema})


def test_public_document_and_route_coverage(app):
    client = app.test_client()
    result = client.get("/openapi.json")
    assert result.status_code == 200
    assert result.mimetype == "application/json"
    assert result.json == DOCUMENT
    assert client.get("/api/config/node-id").status_code == 401
    actual = {
        (re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", rule.rule), method.lower())
        for rule in app.url_map.iter_rules()
        for method in rule.methods - {"HEAD", "OPTIONS"}
    }
    documented = {
        (path, method) for path, ops in DOCUMENT["paths"].items() for method in ops
    }
    assert actual == documented


def test_request_schemas_and_authorization_match_routes():
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {"errorMessages", "errorMessage", "requiredMessages"}
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    for file in (ROOT / "fedbiomed_node_api/routes").glob("*.py"):
        for function in ast.parse(file.read_text()).body:
            if not isinstance(function, ast.FunctionDef):
                continue
            decorators = function.decorator_list
            routes = [
                d
                for d in decorators
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "route"
            ]
            for route in routes:
                prefix = "/api/auth" if route.func.value.id == "auth" else "/api"
                path = re.sub(
                    r"<(?:[^:>]+:)?([^>]+)>",
                    r"{\1}",
                    prefix + ast.literal_eval(route.args[0]),
                )
                methods = next(
                    ast.literal_eval(k.value)
                    for k in route.keywords
                    if k.arg == "methods"
                )
                for method in methods:
                    operation = DOCUMENT["paths"][path][method.lower()]
                    admin = any(
                        isinstance(d, ast.Name) and d.id == "admin_required"
                        for d in decorators
                    )
                    assert operation.get("x-admin-required", False) == admin
                    security = operation.get("security", DOCUMENT["security"])
                    if prefix == "/api":
                        assert security == [{"bearerAuth": []}]
                    else:
                        expected = (
                            [{"refreshToken": []}] if path.endswith("/refresh") else []
                        )
                        assert security == expected
                    for decorator in decorators:
                        if (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "validate_request_data"
                        ):
                            name = next(
                                k.value.id
                                for k in decorator.keywords
                                if k.arg == "schema"
                            )
                            assert DOCUMENT["components"]["schemas"][name] == clean(
                                getattr(schemas, name).schema._schema
                            )
                            body = operation["requestBody"]["content"][
                                "application/json"
                            ]["schema"]
                            assert f"#/components/schemas/{name}" in json.dumps(body)


def test_document_examples_and_generated_docs():
    metadata = tomllib.loads((ROOT / "fedbiomed_node_api/pyproject.toml").read_text())
    assert DOCUMENT["info"]["version"] == metadata["project"]["version"]

    def check_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                target = DOCUMENT
                assert value["$ref"].startswith("#/")
                for part in value["$ref"][2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            for item in value.values():
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)

    check_references(DOCUMENT)
    ids = []
    for operations in DOCUMENT["paths"].values():
        for operation in operations.values():
            ids.append(operation["operationId"])
            if "requestBody" in operation:
                media = operation["requestBody"]["content"]["application/json"]
                Draft202012Validator.check_schema(media["schema"])
                if "example" in media:
                    validator(media["schema"]).validate(media["example"])
    assert len(ids) == len(set(ids))
    for schema in DOCUMENT["components"]["schemas"].values():
        Draft202012Validator.check_schema(schema)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_api_docs.py"), "--check"],
        check=True,
    )


def test_login_refresh_and_dataset_examples(app):
    client = app.test_client()
    login = DOCUMENT["paths"]["/api/auth/token/login"]["post"]
    response = client.post(
        "/api/auth/token/login",
        json=login["requestBody"]["content"]["application/json"]["example"],
    )
    assert response.status_code == 200
    validator(
        login["responses"]["200"]["content"]["application/json"]["schema"]
    ).validate(response.json)
    headers = {"Authorization": "Bearer " + response.json["result"]["access_token"]}
    refresh = client.get(
        "/api/auth/token/refresh",
        headers={"Authorization": "Bearer " + response.json["result"]["refresh_token"]},
    )
    assert refresh.status_code == 200
    assert client.get("/api/config/node-id", headers=headers).status_code == 200
    assert (
        client.post("/api/datasets/list", headers=headers, json={}).status_code == 200
    )
    # Exercise the documented admin request bodies against an isolated database.
    create = DOCUMENT["paths"]["/api/admin/users/create"]["post"]
    user = client.post(
        "/api/admin/users/create",
        headers=headers,
        json=create["requestBody"]["content"]["application/json"]["example"],
    )
    assert user.status_code == 201
    removed = client.delete(
        "/api/admin/users/remove",
        headers=headers,
        json={"user_id": user.json["result"]["user_id"]},
    )
    assert removed.status_code == 200
    Path(app.config["DATA_PATH_RW"], "example.csv").write_text(
        "age,value\n10,1\n20,2\n30,3\n"
    )
    create_dataset = DOCUMENT["paths"]["/api/datasets/add"]["post"]
    dataset = client.post(
        "/api/datasets/add",
        headers=headers,
        json=create_dataset["requestBody"]["content"]["application/json"]["example"],
    )
    assert dataset.status_code == 200
    validator(
        create_dataset["responses"]["200"]["content"]["application/json"]["schema"]
    ).validate(dataset.json)
    assert (
        client.post(
            "/api/datasets/remove",
            headers=headers,
            json={"dataset_id": dataset.json["result"]},
        ).status_code
        == 200
    )
