from tinydb import TinyDB, Query
from tinydb.table import Table

from app import app


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

    def table_users(self) -> Table:
        """Method  for selecting TinyDB table containing the datasets.

        Returns:
            A TinyDB `Table` object for this table.
        """
        return self._table('Users')


node_database = NodeDatabase(app.config['NODE_DB_PATH'])
user_database = UserDatabase(app.config['GUI_DB_PATH'])
