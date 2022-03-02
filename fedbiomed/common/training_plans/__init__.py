"""
The `fedbiomed.common.training_plans` module includes training plan classes
that are used for federated training
"""


from ._fedbiosklearn import SGDSkLearnModel
from ._torchnn import TorchTrainingPlan


__all__ = [
    "SGDSkLearnModel",
    "TorchTrainingPlan",
]
