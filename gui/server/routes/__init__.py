from flask import Blueprint


# Create a blue print for `/api` url prefix. The URLS
api = Blueprint('api', 'api', url_prefix='/api')

# Uses api/ prefix for API endpoints
from .config import *
from .datasets import *
from .repository import *