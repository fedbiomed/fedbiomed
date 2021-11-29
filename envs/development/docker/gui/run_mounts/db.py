from app import app
from tinydb import TinyDB

class Database:

    def __init__(self):
        self.default_table = '_default'

    @property
    def db(self):        
        return TinyDB(app.config['NODE_DB_PATH']) 

    def table(self, name :str):
        print(app.config['NODE_DB_PATH'])
        self.db.table(name)


database = Database()