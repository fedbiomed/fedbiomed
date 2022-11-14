"""
to simplify imports from fedbiomed.node.cli_utils
"""

from ._database import (
    add_database,
    dataset_manager,
    delete_all_database,
    delete_database,
)
from ._training_plan_management import (
    approve_training_plan,
    delete_training_plan,
    register_training_plan,
    reject_training_plan,
    tp_security_manager,
    update_training_plan,
    view_training_plan,
)

__all__ = [
    "add_database",
    "dataset_manager",
    "delete_all_database",
    "delete_database",
    "approve_training_plan",
    "delete_training_plan",
    "register_training_plan",
    "reject_training_plan",
    "tp_security_manager",
    "update_training_plan",
    "view_training_plan",
]
