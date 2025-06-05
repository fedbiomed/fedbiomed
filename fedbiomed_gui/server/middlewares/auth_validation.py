from flask import request


from ..helpers.auth_helpers import check_mail_format, check_password_format, get_user_by_email
from ..utils import error


def validate_email_register():
    """Middleware for validating email in register form"""
    req = request.json

    if not check_mail_format(req['email']):
        return error('Wrong email format'), 400

    if get_user_by_email(req['email']):
        return error('Email already present.'), 409

    return None


def validate_password():
    """Middleware for validating password in register and update password actions"""
    req = request.json
    if not check_password_format(req['password']):
        return error(
            'Password should be at least 8 character long, with at least one uppercase letter, one lowercase letter '
            'and one number'), 400

    return None
