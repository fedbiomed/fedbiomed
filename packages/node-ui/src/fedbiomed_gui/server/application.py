"""Compose the Node API with the built React frontend."""

from pathlib import Path

from fedbiomed_node_api.application import create_app as create_api_app
from flask import send_from_directory


def frontend_assets() -> Path:
    """Locate the built frontend: bundled in the package, else in a source checkout."""
    package_assets_location = Path(__file__).resolve().parent.parent / "ui" / "build"
    if package_assets_location.is_dir():
        return package_assets_location
    return Path(__file__).resolve().parents[3] / "ui" / "build"


def create_app(node_root=None, overrides=None, build_dir=None):
    """Create the GUI server using the same API as headless deployments."""
    app = create_api_app(node_root, overrides)

    assets = Path(build_dir) if build_dir is not None else frontend_assets()

    app.static_folder = str(assets)
    app.add_url_rule("/build/<path:filename>", "static", app.send_static_file)

    @app.route("/", defaults={"path": ""}, methods=["GET"])
    @app.route("/<path:path>")
    def index(path):
        if path and (assets / path).is_file():
            return send_from_directory(assets, path)
        return send_from_directory(assets, "index.html")

    return app
