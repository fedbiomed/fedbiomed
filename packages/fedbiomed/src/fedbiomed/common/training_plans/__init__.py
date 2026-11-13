# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
The `fedbiomed.common.training_plans` module includes training plan classes
that are used for federated training
"""

from ._base_training_plan import BaseTrainingPlan
from ._sklearn_models import FedPerceptron, FedSGDClassifier, FedSGDRegressor
from ._sklearn_training_plan import SKLearnTrainingPlan
from ._torchnn import TorchTrainingPlan

__all__ = [
    "TorchTrainingPlan",
    "SKLearnTrainingPlan",
    "FedPerceptron",
    "FedSGDRegressor",
    "FedSGDClassifier",
    "BaseTrainingPlan",
]
