"""
The `fedbiomed.common.training_plans` module includes training plan classes
that are used for federated training
"""

from .fedbiosklearn import SGDSkLearnModel
from .torchnn import TorchTrainingPlan


__all__ = [
    "SGDSkLearnModel",
    "TorchTrainingPlan",
]