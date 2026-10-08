"""Check an isolated release installation, outside the source checkout."""

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def verify(profile, workspace, expected_python, dist_dir):
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == expected_python
    from fedbiomed.common.utils import SHARE_DIR

    expected = {"fedbiomed"}
    # Check both presence and absence; installing every extra together would
    # conceal accidental coupling between the core, API, and GUI packages.
    if profile in {"node-api", "gui"}:
        expected.add("fedbiomed-node-api")
    if profile == "gui":
        expected.add("fedbiomed-gui")
    core = distribution("fedbiomed")
    assert {"node", "node-api", "gui", "researcher"} <= set(
        core.metadata.get_all("Provides-Extra")
    )
    assert core.metadata["Requires-Python"]
    for name in ("fedbiomed", "fedbiomed-node-api", "fedbiomed-gui"):
        module = name.replace("-", "_")
        if name not in expected:
            assert importlib.util.find_spec(module) is None, (
                f"Unexpected package: {name}"
            )
            try:
                distribution(name)
            except PackageNotFoundError:
                continue
            raise AssertionError(f"Unexpected distribution: {name}")
        package = distribution(name)
        assert package.version == core.version
        direct = json.loads(package.read_text("direct_url.json"))
        # Matching versions alone do not prove that pip used our built wheel:
        # the same version might already exist on an index or in the checkout.
        wheel = dist_dir / f"{module}-{package.version}-py3-none-any.whl"
        assert direct["url"] == wheel.as_uri(), (
            f"Did not install release artifact: {name}"
        )
        loaded = importlib.import_module(module)
        assert not Path(loaded.__file__).resolve().is_relative_to(workspace)
        commands = {
            entry.name: entry.value
            for entry in package.entry_points
            if entry.group == "console_scripts"
        }
        assert commands[name] == f"{module}.cli:run"
        subprocess.run([str(Path(sys.executable).parent / name), "--help"], check=True)
    if profile in {"core", "researcher"}:
        for module in ("flask", "flask_jwt_extended", "cachelib", "gunicorn"):
            assert importlib.util.find_spec(module) is None, (
                f"Unexpected web dependency: {module}"
            )
    if profile == "core":
        assert importlib.util.find_spec("jsonschema") is None
    if profile == "researcher":
        importlib.import_module("fedbiomed.researcher.experiment")
    for relative in ("notebooks", "docs/tutorials", "envs/common"):
        # These resources are installed under SHARE_DIR rather than the module.
        directory = Path(SHARE_DIR) / relative
        assert directory.is_dir() and any(directory.rglob("*")), directory
    if profile not in {"node-api", "gui"}:
        print(f"{profile}: installed package boundaries and resources verified")
        return

    from fedbiomed.node.config import node_component
    from fedbiomed_node_api.application import create_app

    with tempfile.TemporaryDirectory(prefix="fbm-installed-api-") as tmp:
        # Use disposable node state and Flask's in-process client. This tests
        # installed application routes/resources, not a live Gunicorn process.
        root = Path(tmp) / "node"
        node_component.initiate(root=str(root))
        os.environ["DATA_PATH"] = str(root / "data")
        if profile == "gui":
            from fedbiomed_gui.server.application import create_app
        app = create_app(root, {"TESTING": True})
        client = app.test_client()
        # The contract must ship with the wheel and be accessible without a token,
        # including when the GUI adds its catch-all frontend route.
        reference = client.get("/openapi.json")
        assert reference.status_code == 200
        assert reference.mimetype == "application/json"
        assert (
            reference.json["info"]["version"]
            == distribution("fedbiomed-node-api").version
        )
        assert "/api/auth/token/login" in reference.json["paths"]
        assert client.get("/api/config/node-id").status_code == 401
        # Authenticate through the real login route, then reuse its issued token.
        login = client.post(
            "/api/auth/token/login",
            json={"email": "admin@fedbiomed.gui", "password": "admin"},
        )
        assert login.status_code == 200
        token = login.json["result"]["access_token"]
        response = client.get(
            "/api/config/node-id", headers={"Authorization": "Bearer " + token}
        )
        assert (
            response.status_code == 200
            and response.json["result"]["node_id"] == app.config["ID"]
        )
        assert client.get("/").status_code == (200 if profile == "gui" else 404)
        if profile == "gui":
            # Check actual served bytes, not just status 200 or file existence.
            assets = Path(app.static_folder)
            assert client.get("/").data == (assets / "index.html").read_bytes()
            js = next(assets.rglob("*.js"))
            assert (
                client.get("/" + js.relative_to(assets).as_posix()).data
                == js.read_bytes()
            )
        else:
            assert app.static_folder is None
            assert not any(name.startswith("fedbiomed_gui") for name in sys.modules)
    print(f"{profile}: installed artifacts, authentication and serving verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["core", "node-api", "gui", "researcher"], required=True
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.profile,
        args.workspace.resolve(),
        args.expected_python,
        args.dist_dir.resolve(),
    )
