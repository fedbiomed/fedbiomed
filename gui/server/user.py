from hashlib import sha512
from xmlrpc.client import Boolean
from flask_login import UserMixin, LoginManager
from datetime import datetime
from db import gui_database
from fedbiomed.common.constants import UserRoleType
from app import login_manager


class User:

    def __init__(self, UserMixin) -> None:
        """ User account class for authentication. 
            Uses flask_login.
        """
        self._user_id = None
        self._user_email = None
        self._creation_date = None
        self._password_hash = None
        self._user_role = None

    def get_user(self):
        return self

    def get_user_email(self):
        return self._user_email

    def get_creation_date(self):
        return self._creation_date

    def get_password_hash(self):
        return self._password_hash

    def get_user_role(self):
        return self._user_role

    def set_user_email(self, email: str):
        # TODO : check email format
        self._user_email = email

    def set_creation_date(self, date: datetime):
        self._creation_date = date

    def set_user_role(self, role: UserRoleType):
        self._user_role = role

    def set_password_hash(self, password: str) -> str:
        """ Method for setting password hash 
        Args: 

            password    (str): Password of the user
        """
        self._password_hash = sha512(password.encode)

    def check_password_hash(self, password: str) -> Boolean:
        """ Method used to compare password hashes. 
            Used to verify the user password
        Args: 

            password (str): Password to compare against the user password hash
        Returns:
            True if the password hash matches the user password one
            Flase otherwise
        """
        password_hash = sha512(password.encode)
        return password_hash.digest() == self._password_hash.digest()


@login_manager.user_loader
def load_user(user_id: str):
    """ Method used to ling the database and the User ID since
        Flask stores the user id of the logged-in users in the session
    Args: 

        user_id (str): The id of the user to retrieve from the database
    """
    # TODO: Use specific user table in gui_database
    table = gui_database.db().table('_default')
    query = gui_database.query()
    return table.get(query.user_id == user_id)
