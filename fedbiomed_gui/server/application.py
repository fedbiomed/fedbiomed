"""Compose the Node API with the built React frontend."""

from pathlib import Path

from flask import send_from_directory

from fedbiomed_gui.cli import assets_directory, require_assets
from fedbiomed_node_api.application import create_app as create_api_app


def create_app(node_root=None, overrides=None, build_dir=None):
    """Create the GUI server using the same API as headless deployments."""
    assets = Path(build_dir) if build_dir is not None else assets_directory()
    require_assets(assets)
    app = create_api_app(node_root, overrides)
    app.static_folder = str(assets)
    app.add_url_rule("/build/<path:filename>", "static", app.send_static_file)

    @app.route("/", defaults={"path": ""}, methods=["GET"])
    @app.route("/<path:path>")
    def index(path):
        if path and (assets / path).is_file():
            return send_from_directory(assets, path)
        return send_from_directory(assets, "index.html")

    return app
