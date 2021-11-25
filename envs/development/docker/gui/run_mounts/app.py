import sys
from flask import Flask, request, abort
from tinydb import TinyDB

sys.path.append('/fedbiomed')

app = Flask(__name__)
database = TinyDB('/fedbiomed/var/db_node_0e284192-5648-4a99-9074-2727876dec75.json')

#  Import uploads routes
import routes.upload 

  
