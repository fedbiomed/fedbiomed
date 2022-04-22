import sys
import numpy as np

from typing import Any, Dict, Union, Callable
from io import StringIO
from joblib import dump, load

from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from ._base_training_plan import BaseTrainingPlan

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans, ProcessTypes
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.utils import get_method_spec

from ._sklearn_training_plan import SKLearnTrainingPlan

class FedPerceptron(SKLearnTrainingPlan):


    def __init__(self, model_args):

        super().__init__(Perceptron,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = True)

        # specific for Perceptron
        self.is_classification = True


    def training_routine_hook(self):
        (data, target) = self.training_data_loader
        classes = self.__classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(data, target, classes=classes)


#======

class FedSGDRegressor(SKLearnTrainingPlan):


    def __init__(self, model_args):

        super().__init__(SGDRegressor,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = True)

        # specific for SGDRegressor
        self._is_regression = True

    def training_routine_hook(self):
        (data, target) = self.training_data_loader
        self.model.partial_fit(data, target)


#======

class FedMultinomialNB(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        Raise("model not implemented yet")

    def training_routine_hook(self):
        pass
