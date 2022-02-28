"""
The `fedbiomed.common.training_plans` module includes training plan classes
that are used for federated training
"""

from .base_training_plan import BaseTrainingPlan
from .fedbiosklearn import SGDSkLearnModel
from .torchnn import TorchTrainingPlan


__all__ = [
    "BaseTrainingPlan",
    "SGDSkLearnModel",
    "TorchTrainingPlan",
]