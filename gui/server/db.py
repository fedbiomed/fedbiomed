from app import app
from tinydb import TinyDB, Query
from tinydb.table import Table

class Database:

    def __init__(self):
        """ Database class for TinyDB. It is general wrapper for
            TinyDB. It can be extended in the future, if Fed-BioMed
            support a=other persistent databases.
        """
        self._db = None
        self._query = Query()

    def db(self):
        """ Getter for db """
        self._db = TinyDB(app.config['NODE_DB_PATH'])

        return self

    def query(self):
        return self._query

    def table_datasets(self) -> Table:
        """Method  for selecting TinyDB table containing the datasets.

        Returns:
            A TinyDB `Table` object for this table. 
        """
        return self._table('Datasets')

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

    def close(self):
        """This method removes TinyDB object to save some memory"""

        self._db.close()


database = Database()
