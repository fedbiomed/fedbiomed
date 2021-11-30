from tinydb import TinyDB, Query
class Database:

    def __init__(self):
        self._db = None
        self._query = None

    @property
    def db(self): 
        self._db = TinyDB(app.config['NODE_DB_PATH'])      
        return self._db 

    @property
    def query(self):
        if self._db:
            return self._query
        else:
            raise Exception('Please initialize database first with database.db')

    def table(self, name : str = '_default'):
        """ Method for selecting table 

        Args: 

            name    (str): Table name. Default is `_default`
                            when there is no table seficified TinyDB
                            write data into `_dafualt` table 
        """

        return self.db.table(name)

    def close(self):

        """This method remove TinyDB object to save some memory"""
        
        self.__dict__.pop('_db',None)
    



database = Database()