from flask import request

from ..services import service_proxy
from ..utils import error

# Initialize Fed-BioMed DatasetManager
dataset_manager = service_proxy("dataset_manager")


def check_tags_already_registered():
    """Middleware that checks requested tags is already existing"""
    req = request.json
    tags = req["tags"]

    conflicting = dataset_manager.dataset_table.search_conflicting_tags(tags)
    if len(conflicting) > 0:
        return error(
            "one or more datasets are already registered with conflicting tags: "
            f"{' '.join([c['name'] for c in conflicting])}"
        ), 400
