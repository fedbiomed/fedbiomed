from datetime import datetime
from typing import Dict
import uuid

from fedbiomed.common.constants import UserRoleType
from tinydb import TinyDB, Query
from tinydb.table import Table

from .config import config
from .utils import set_password_hash


# WARNING: this Database class should not exist, all accesses to TinyDB should occur
# through the fedbiomed.node_dataset_manager.DatasetManager, this
# breaks basic object paradigm
# Until this is refactored (and Database class is removed), please use
# `DatasetManager` when adding new accesses to the database

class BaseDatabase:

    def __init__(self, db_path: str):
        """ Database class for TinyDB. It is general wrapper for
            TinyDB. It can be extended in the future, if Fed-BioMed
            support a=other persistent databases.
        """
        self._db = TinyDB(db_path)
        self._query = Query()

    def query(self):
        return self._query

    def _table(self, name: str) -> Table:
        """ Method for selecting table

        Args:

            name    (str): Table name.

        Returns:
            A TinyDB `Table` object for the selected table.
        """

        if self._db is None:
            raise Exception('Please initialize database first')

        # don't use read cache to avoid coherence problems
        return self._db.table(name=name, cache_size=0)


class NodeDatabase(BaseDatabase):

    def __init__(self, db_path: str):
        super(NodeDatabase, self).__init__(db_path)

    def table_datasets(self) -> Table:
        """Method  for selecting TinyDB table containing the datasets.

        Returns:
            A TinyDB `Table` object for this table.
        """
        return self._table('Datasets')


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
        email, password = admin_credential['email'], admin_credential['password']

        # first step: check if there is no admin registered in database
        try:
            query = self.query()
            admins = self.table('Users').get(query.user_role == UserRoleType.ADMIN)
            if not admins:
                # if no admin user are found, add it into user gui database
                print("No admin found, creating default one")
                self.table('Users').insert(
                    {"user_email": email,
                     "password_hash": set_password_hash(password),
                     "user_name": "System",
                     "user_surname": "Admin",
                     "user_role": UserRoleType.ADMIN,
                     "creation_date": datetime.utcnow().ctime(),
                     "user_id": 'user_' + str(uuid.uuid4())})
        except Exception as e:
            print(f"Error, unable to query in database for admin accounts {e}... resuming")


node_database = NodeDatabase(config['NODE_DB_PATH'])
user_database = UserDatabase(config['GUI_DB_PATH'])

user_database.add_default_admin_user(config['DEFAULT_ADMIN_CREDENTIAL'])
# remove default account credential of env variables
# for security reasons
del config['DEFAULT_ADMIN_CREDENTIAL']
