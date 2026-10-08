"""Create the Node HTTP API without importing or serving the GUI."""

import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, send_file
from flask_jwt_extended import JWTManager

from .config import Config
from .services import init_services
from .utils import error


def create_app(node_root=None, overrides=None):
    """Create an API for a node root (or ``FBM_NODE_COMPONENT_ROOT``).

    Configuration and services belong to this application. Importing this module
    does not open a node database or require frontend assets.
    """
    app = Flask(__name__, static_folder=None)
    config = Config(node_root)
    config.configuration.update(overrides or {})
    app.extensions["node_api_config"] = config
    init_services(app, config)
    app.config.update(config.configuration)
    app.config.update(
        JWT_TOKEN_LOCATION=["headers"],
        JWT_COOKIE_SECURE=True,
        JWT_ACCESS_TOKEN_EXPIRES=timedelta(minutes=30),
        JWT_REFRESH_TOKEN_EXPIRES=timedelta(minutes=60),
        JWT_COOKIE_CSRF_PROTECT=True,
        JWT_ENCODE_ISSUER=config["ID"],
        JWT_DECODE_ISSUER=config["ID"],
        SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(),
    )
    app.config.update(overrides or {})
    app.config.pop("DEFAULT_ADMIN_CREDENTIAL", None)
    assert (
        app.config["JWT_ACCESS_TOKEN_EXPIRES"] < app.config["JWT_REFRESH_TOKEN_EXPIRES"]
    )

    jwt = JWTManager(app)

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return error(msg="Session has expired! Please login again"), 401

    from .routes import api, auth

    app.register_blueprint(api)
    app.register_blueprint(auth)

    # Public, static API contract: available without login or GUI assets.
    # Keep it outside the /api blueprint, whose before_request requires a JWT.
    @app.get("/openapi.json")
    def openapi_document():
        return send_file(
            Path(__file__).with_name("openapi.json"), mimetype="application/json"
        )

    return app
