"""Storage for API user accounts and registration requests.

Node dataset storage is managed by the core DatasetManager service.
"""

import uuid
from datetime import datetime
from typing import Dict

from flask import current_app
from tinydb import Query, TinyDB
from tinydb.table import Table
from werkzeug.local import LocalProxy

from fedbiomed.common.constants import UserRoleType

from .utils import set_password_hash


class BaseDatabase:
    def __init__(self, db_path: str):
        """Database class for TinyDB. It is general wrapper for
        TinyDB. It can be extended in the future, if Fed-BioMed
        support a=other persistent databases.
        """
        self._db = TinyDB(db_path)
        self._query = Query()

    def query(self):
        return self._query

    def _table(self, name: str) -> Table:
        """Method for selecting table

        Args:

            name    (str): Table name.

        Returns:
            A TinyDB `Table` object for the selected table.
        """

        if self._db is None:
            raise Exception("Please initialize database first")

        # don't use read cache to avoid coherence problems
        return self._db.table(name=name, cache_size=0)


class UserDatabase(BaseDatabase):
    def __init__(self, db_path: str):
        super(UserDatabase, self).__init__(db_path)

    def table(self, table_name: str) -> Table:
        """Method  for selecting TinyDB table named table_name.

        Returns:
            A TinyDB `Table` object for this table.
        """
        return self._table(table_name)

    def add_default_admin_user(self, admin_credential: Dict[str, str]):
        """adds default admin user to database if no admin has been found in database"""
        email, password = admin_credential["email"], admin_credential["password"]

        # first step: check if there is no admin registered in database
        try:
            query = self.query()
            admins = self.table("Users").get(query.user_role == UserRoleType.ADMIN)
            if not admins:
                # if no admin user are found, add it into user gui database
                print("No admin found, creating default one")
                self.table("Users").insert(
                    {
                        "user_email": email,
                        "password_hash": set_password_hash(password),
                        "user_name": "System",
                        "user_surname": "Admin",
                        "user_role": UserRoleType.ADMIN,
                        "creation_date": datetime.utcnow().ctime(),
                        "user_id": "user_" + str(uuid.uuid4()),
                    }
                )
        except Exception as e:
            print(
                f"Error, unable to query in database for admin accounts {e}... resuming"
            )


user_database = LocalProxy(
    lambda: current_app.extensions["node_api_services"]["user_database"]
)
