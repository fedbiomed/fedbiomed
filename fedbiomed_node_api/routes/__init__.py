# Uses api/ prefix for API endpoints
# Submodules are imported for their side effect: registering handlers on the blueprints.
from . import (
    authentication,
    certificates,
    config,
    datasets,
    medical_folder_dataset,
    model_router,
    node_management,
    repository,
    security_logs,
    users,
)
from .api import api, auth

__all__ = [
    "api",
    "auth",
    "authentication",
    "certificates",
    "config",
    "datasets",
    "medical_folder_dataset",
    "model_router",
    "node_management",
    "repository",
    "security_logs",
    "users",
]
