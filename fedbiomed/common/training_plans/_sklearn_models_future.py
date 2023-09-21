# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Yet-to-be-implemented TrainingPlan for additional scikit-learn models."""

import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger

from ._sklearn_training_plan import SKLearnTrainingPlan


class FedBernoulliNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of BernoulliNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    _model_cls = BernoulliNB

    def __init__(self):
        """Sklearn BernoulliNB model.

        Args:
            model_args: model arguments. Defaults to {}
        """

        msg = ErrorNumbers.FB605.value + \
            " FedBernoulliNB not implemented."
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)
        super().__init__()

        self.is_classification = True
        self._add_dependency([
            "from sklearn.naive_bayes import BernoulliNB"
        ])

    def training_routine_hook(self):
        """Training routine of BernoulliNB.
        """
        data, target = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self._model.partial_fit(data, target, classes=classes)

    def set_init_params(self):
        """Initialize the model parameter.
        """
        if 'verbose' in self._model_args:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for BernoulliNB "
                         ": it needs to be implemented")


class FedGaussianNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of FedGaussianNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    _model_cls = GaussianNB

    def __init__(self):
        """
        Sklearn GaussianNB model.

        Args:
            model_args: model arguments. Defaults to {}
        """

        msg = ErrorNumbers.FB605.value + \
            " FedGaussianNB not implemented."
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

        super().__init__()

        self.is_classification = True
        self._add_dependency([
            "from sklearn.naive_bayes  import GaussianNB"
        ])

    def training_routine_hook(self):
        """Training routine of GaussianNB.
        """
        data, target = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self._model.partial_fit(data, target, classes=classes)

    def set_init_params(self):
        """Initialize the model parameter.
        """

        if 'verbose' in self._model_args:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for GaussianNB "
                         ": it needs to be implemented")

        self._param_list = ['intercept_', 'coef_']
        init_params = {
            'intercept_': np.array([0.]) if (self._model_args['n_classes'] == 2) else np.array(
                [0.] * self._model_args['n_classes']),
            'coef_': np.array([0.] * self._model_args['n_features']).reshape(1, self._model_args['n_features']) if (
                self._model_args['n_classes'] == 2) else np.array(
                    [0.] * self._model_args['n_classes'] * self._model_args['n_features']).reshape(
                        self._model_args['n_classes'],
                        self._model_args['n_features'])
        }

        for p in self._param_list:
            setattr(self._model, p, init_params[p])


class FedMultinomialNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of FedMultinomialNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self):
        msg = ErrorNumbers.FB605.value + \
            " FedMultinomialNB not implemented."
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass


class FedPassiveAggressiveClassifier(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of PassiveAggressiveClassifier class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self):
        msg = ErrorNumbers.FB605.value + \
            ": model FedPassiveAggressiveClassifier not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass


class FedPassiveAggressiveRegressor(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of class PassiveAggressiveRegressor from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self):
        msg = ErrorNumbers.FB605.value + \
            ": model FedPassiveAggressiveRegressor not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass


class FedMiniBatchKMeans(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of MiniBatchKMeans class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self):
        msg = ErrorNumbers.FB605.value + \
            ": model FedMiniBatchKMeans not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass


class FedMiniBatchDictionaryLearning(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of MiniBatchDictionaryLearning class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self):
        msg = ErrorNumbers.FB605.value + \
            ": model FedMiniBatchDictionaryLearning not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass
