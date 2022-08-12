from db import node_database
from flask import request
from utils import error


def check_tags_already_registered():
    """Middleware that checks requested tags is already existing"""
    req = request.json
    tags = req["tags"]
    table = node_database.table_datasets()
    query = node_database.query()

    table.clear_cache()
    found = table.search(query.tags.all(tags))

    if len(found) > 0:
        return error(f'There is already a dataset added with the same tags, {req["tags"]}'), 400
