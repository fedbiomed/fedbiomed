# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
to simplify imports from fedbiomed.node.cli_utils
"""

from ._database import add_database, delete_database, delete_all_database
from ._training_plan_management import register_training_plan, update_training_plan, approve_training_plan, reject_training_plan, \
    delete_training_plan, view_training_plan

__all__ = [
    'add_database',
    'delete_database',
    'delete_all_database',
    'register_training_plan',
    'update_training_plan',
    'approve_training_plan',
    'reject_training_plan',
    'delete_training_plan',
    'view_training_plan'
]
