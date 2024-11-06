from flask import request

from fedbiomed.node.dataset_manager import DatasetManager

from ..utils import error

# Initialize Fed-BioMed DatasetManager
dataset_manager = DatasetManager()


def check_tags_already_registered():
    """Middleware that checks requested tags is already existing"""
    req = request.json
    tags = req["tags"]

    conflicting = dataset_manager.search_conflicting_tags(tags)
    if len(conflicting) > 0:
        return error("one or more datasets are already registered with conflicting tags: "
                     f"{' '.join([ c['name'] for c in conflicting ])}"), 400
