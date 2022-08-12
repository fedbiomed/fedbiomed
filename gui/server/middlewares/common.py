from flask import request, g
from db import database
from utils import error


def check_tags_already_registered():
    """Middleware that checks requested tags is already existing"""
    req = request.json
    tags = req["tags"]
    table = database.db().table_datasets()
    query = database.query()

    found = table.search(query.tags.all(tags))

    if len(found) > 0:
        return error(f'There is already a dataset added with the same tags, {req["tags"]}'), 400
