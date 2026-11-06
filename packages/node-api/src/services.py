"""Node services owned by an API application, resolved within its context."""

from flask import current_app
from werkzeug.local import LocalProxy


def service_proxy(name):
    """Resolve a service for the current application without import-time I/O."""
    return LocalProxy(lambda: current_app.extensions["node_api_services"][name])


def init_services(app, config):
    from pathlib import Path

    from cachelib import FileSystemCache

    from fedbiomed.node.dataset_manager import DatasetManager
    from fedbiomed.node.node_pm import NodeProcessManager
    from fedbiomed.node.training_plan_security_manager import (
        TrainingPlanSecurityManager,
    )

    from .db import UserDatabase

    users = UserDatabase(config["GUI_DB_PATH"])
    users.add_default_admin_user(config.configuration.pop("DEFAULT_ADMIN_CREDENTIAL"))
    app.extensions["node_api_services"] = {
        "user_database": users,
        "dataset_manager": DatasetManager(config["NODE_DB_PATH"]),
        "node_process_manager": NodeProcessManager(config.node_config),
        "training_plan_manager": TrainingPlanSecurityManager(
            db=config["NODE_DB_PATH"],
            node_name=config.node_config.get("default", "name"),
            node_id=config["ID"],
            hashing=config.node_config.get("security", "hashing_algorithm"),
            tp_approval=config.node_config.getbool(
                "security", "training_plan_approval"
            ),
        ),
        "cache": FileSystemCache(str(Path(config["GUI_DB_PATH"]).parent / "api_cache")),
        "repository_file_sizes": {},
    }
