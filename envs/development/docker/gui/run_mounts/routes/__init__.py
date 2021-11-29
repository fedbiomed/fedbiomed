from flask import Blueprint


# Create a blue print for `/api` url prefix. The URLS
api = Blueprint('api', 'api', url_prefix='/api')


# Uses default `/` prefix for index and static
from .index import *

# Uses `/api` prefix 
from .list import *
from .upload import *