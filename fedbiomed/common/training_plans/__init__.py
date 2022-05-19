"""
The `fedbiomed.common.training_plans` module includes training plan classes
that are used for federated training
"""


from ._fedbiosklearn import SGDSkLearnModel
from ._torchnn import TorchTrainingPlan
from ._sklearn_training_plan import SKLearnTrainingPlan
from ._sklearn_models import FedPerceptron, FedSGDRegressor, FedSGDClassifier
from ._base_training_plan import BaseTrainingPlan

__all__ = [
    "SGDSkLearnModel",
    "TorchTrainingPlan",
    "SKLearnTrainingPlan",
    "FedPerceptron",
    "FedSGDRegressor",
    'FedSGDClassifier',
    "BaseTrainingPlan"
]
