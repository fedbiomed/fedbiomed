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
        self.set_init_params(model_args)

    def training_routine_hook(self):
        (data, target) = self.training_data_loader
        classes = self.__classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(data, target, classes=classes)

    def set_init_params(self,model_args):
        self.param_list = ['intercept_','coef_']
        init_params = {
            'intercept_': np.array([0.]) if (model_args['n_classes'] == 2) else np.array(
                [0.] * model_args['n_classes']),
            'coef_': np.array([0.] * model_args['n_features']).reshape(1, model_args['n_features']) if (
                    model_args['n_classes'] == 2) else np.array(
                [0.] * model_args['n_classes'] * model_args['n_features']).reshape(model_args['n_classes'],
                                                                                   model_args['n_features'])
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params_sgd:
            setattr(self.model, p, self.params_sgd[p])


#======

class FedSGDRegressor(SKLearnTrainingPlan):


    def __init__(self, model_args):

        super().__init__(SGDRegressor,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = True)

        # specific for SGDRegressor
        self._is_regression = True
        self.set_init_params(model_args)

    def training_routine_hook(self):
        (data, target) = self.training_data_loader
        self.model.partial_fit(data, target)

    def set_init_params(self,model_args):
        self.param_list = ['intercept_','coef_']
        init_params = {'intercept_': np.array([0.]),
                       'coef_': np.array([0.] * model_args['n_features'])}
        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params_sgd:
            setattr(self.model, p, self.params_sgd[p])



#======

class FedSGDClassifier(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):
        super().__init__(SGDClassifier,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = True)

        self.is_classification = True
        self.set_init_params(model_args)

    def training_routine_hook(self):
        pass

    def set_init_params(self,model_args):
        self.param_list = ['intercept_','coef_']
        init_params = {
            'intercept_': np.array([0.]) if (model_args['n_classes'] == 2) else np.array(
                [0.] * model_args['n_classes']),
            'coef_': np.array([0.] * model_args['n_features']).reshape(1, model_args['n_features']) if (
                    model_args['n_classes'] == 2) else np.array(
                [0.] * model_args['n_classes'] * model_args['n_features']).reshape(model_args['n_classes'],
                                                                                   model_args['n_features'])
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params_sgd:
            setattr(self.model, p, self.params_sgd[p])

class FedBernoulliNB(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):
        super().__init__(BernoulliNB,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = False)

        self.is_classification = True

    def training_routine_hook(self):
        pass

class FedGaussianNB(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):
        super().__init__(GaussianNB,
                         model_args,
                         self.training_routine_hook,
                         verbose_possibility = False)

    def training_routine_hook(self):
        pass



#############################################################################################3
class FedMultinomialNB(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass

class FedPassiveAggressiveClassifier(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass

class FedPassiveAggressiveRegressor(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass

class FedMiniBatchKMeans(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass

class FedMiniBatchDictionaryLearning(SKLearnTrainingPlan):

    #
    # or simply do not provide this file
    #

    def __init__(self, model_args):

        raise("model not implemented yet")

    def training_routine_hook(self):
        pass