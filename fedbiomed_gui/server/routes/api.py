from flask import Blueprint, request
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request

from ..helpers.auth_helpers import password_change_required, query, user_table
from ..utils import error

# Create a blue print for `/api` url prefix. The URLS
api = Blueprint("api", __name__, url_prefix="/api")
auth = Blueprint("auth", __name__, url_prefix="/api/auth")


@api.before_request
def before_api_request():
    try:
        verify_jwt_in_request()
    except Exception:
        return error("Invalid token"), 401

    user = user_table.get(query.user_id == get_jwt_identity())
    if not user:
        return error("Invalid user"), 401
    if password_change_required(user) and request.endpoint not in {
        "api.update_password",
        "api.auto_auth",
        "api.logout",
    }:
        return error("You must change your password before continuing"), 403
