"""Node services owned by an API application, resolved within its context."""

from pathlib import Path

from cachelib import FileSystemCache
from flask import current_app
from werkzeug.local import LocalProxy

from fedbiomed.common.db import session_factory
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.node_pm import NodeProcessManager
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager


def service_proxy(name):
    """Resolve a service for the current application without import-time I/O."""
    return LocalProxy(lambda: current_app.extensions["node_api_services"][name])


def init_services(app, config):
    # `.db` imports `service_proxy` from this module
    from .db import UserDatabase

    users = UserDatabase(config["GUI_DB_PATH"])
    users.add_default_admin_user(config.configuration.pop("DEFAULT_ADMIN_CREDENTIAL"))
    sessions = session_factory(
        str(Path(config["NODE_DB_PATH"]).with_suffix(".sqlite")),
        [*DatasetManager.MODELS],
    )
    app.extensions["node_api_services"] = {
        "user_database": users,
        "dataset_manager": DatasetManager(sessions),
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
