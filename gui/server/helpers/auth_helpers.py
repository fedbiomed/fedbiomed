import re
from hashlib import sha512
from flask import request
from functools import wraps
from flask_jwt_extended import (jwt_required, create_access_token, create_refresh_token, unset_jwt_cookies,
                                verify_jwt_in_request, get_jwt)

from fedbiomed.common.constants import UserRoleType

from ..db import user_database
from ..utils import error, response

user_table = user_database.table('Users')
query = user_database.query()


def set_password_hash(password: str) -> str:
    """ Method for setting password hash
    Args:

        password (str): Password of the user
    """
    return sha512(password.encode('utf-8')).hexdigest()


def check_password_hash(password: str, user_password_hash: str) -> bool:
    """ Method used to compare password hashes.
        Used to verify the user password
    Args:

        password (str): Password to compare against the user password hash
        user_password_hash (str): User password hash
    Returns:
        True if the password hash matches the user password one
        False otherwise
    """
    password_hash = sha512(password.encode('utf-8'))
    return password_hash.hexdigest() == user_password_hash


def get_user_by_email(user_email: str) -> str:
    """ Method used to retrieve a user from the database based on its email
    Args:

        user_email (str): The mail of the user to retrieve from the database
    """
    return user_table.get(query.user_email == user_email)


def check_mail_format(user_mail: str) -> bool:
    """ Method used to check the format of the user email
    Args:

        user_mail (str): The mail to check
    """
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, user_mail)


def check_password_format(user_password: str) -> bool:
    """ Method used to check the format of the user password
    Args:

        user_password (str): The password to check. It should be
        - at least 8 character long
        - with at least one uppercase letter, one lowercase letter and one number
    """
    regex = r'^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9]).{8,}$'
    return re.fullmatch(regex, user_password)


def admin_required(func):
    """Decorator used to protect endpoints that require admin role"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()

        user_from_db = user_table.get(query.user_id == claims["sub"])

        if not user_from_db:
            return error("Can not check user role. Please contact system provider"), 400

        if claims['role'] != UserRoleType.ADMIN and user_from_db["user_role"] == UserRoleType.ADMIN:
            return error("You don't have permission to perform this action! Please contact your "
                         "local Administrator"), 403
        else:
            return func(*args, **kwargs)

    return wrapper
