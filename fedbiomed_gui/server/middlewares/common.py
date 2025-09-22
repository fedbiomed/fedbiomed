from flask import request

from fedbiomed.node.dataset_manager import DatasetDatabaseManager

from ..config import config
from ..utils import error

# Initialize Fed-BioMed DatasetManager
dataset_manager = DatasetDatabaseManager(config["NODE_DB_PATH"])


def check_tags_already_registered():
    """Middleware that checks requested tags is already existing"""
    req = request.json
    tags = req["tags"]

    conflicting = dataset_manager.search_conflicting_datasets_by_tags(tags)
    if len(conflicting) > 0:
        return error(
            "one or more datasets are already registered with conflicting tags: "
            f"{' '.join([c['name'] for c in conflicting])}"
        ), 400
