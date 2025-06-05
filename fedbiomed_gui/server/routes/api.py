from flask import Blueprint
from flask_jwt_extended import verify_jwt_in_request

from ..utils import error

# Create a blue print for `/api` url prefix. The URLS
api = Blueprint('api', __name__, url_prefix='/api')
auth = Blueprint('auth', __name__, url_prefix='/api/auth')


@api.before_request
def before_api_request():
    try:
        verify_jwt_in_request()
    except Exception as e:
        return error('Invalid token'), 401



