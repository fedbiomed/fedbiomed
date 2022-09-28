"""
Fed-BioMed encapsulation of SKLearn base classes.
"""


from typing import List
import numpy as np
from io import StringIO

from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger

from ._sklearn_training_plan import SKLearnTrainingPlan


class FedSGDRegressor(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of SGDRegressor class from scikit-learn.

    Attributes:
        _model_cls: an instance of the sklearn class that we are wrapping (static attribute).
    """

    _model_cls = SGDRegressor

    def __init__(self):
        """ Sklearn SGDRegressor model. """

        super().__init__()

        # specific for SGDRegressor
        self._is_regression = True
        self.add_dependency([
            "from sklearn.linear_model import SGDRegressor ",
            "from fedbiomed.common.training_plans import FedSGDRegressor"
        ])

    def training_routine_hook(self) -> None:
        """
        Training routine of SGDRegressor.
        """
        (self.data, self.target) = self.training_data_loader
        self._model.partial_fit(self.data, self.target)

    def set_init_params(self) -> None:
        """
        Initialize the model parameter.
        """
        if 'verbose' not in self._model_args:
            self._model_args['verbose'] = 1
            self._params.update({'verbose': 1})
        self._verbose_capture_option = self._model_args["verbose"]

        self._param_list = ['intercept_', 'coef_']
        init_params = {'intercept_': np.array([0.]),
                       'coef_': np.array([0.] * self._model_args['n_features'])}

        for p in self._param_list:
            setattr(self._model, p, init_params[p])

        for p in self._params:
            setattr(self._model, p, self._params[p])
            
    def get_learning_rate(self) -> List[float]:
        _lr_key = 'eta0'
        return super().get_learning_rate(_lr_key)

    def evaluate_loss(self, output: StringIO, epoch: int) -> float:
        """
        Evaluate the loss.

        Args:
            output: output of the scikit-learn SGDRegressor model during training
            epoch: epoch number

        Returns:
            float: the loss captured in the output
        """

        _loss_collector = self._evaluate_loss_core(output, epoch)
        loss = _loss_collector[-1]
        return loss


class FedSGDClassifier(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of SGDClassifier class from scikit-learn.

    Attributes:
        _model_cls: an instance of the sklearn class that we are wrapping (static attribute).
    """

    _model_cls = SGDClassifier

    def __init__(self):
        """
        Sklearn SGDClassifier model.

        Args:
            model_args: model arguments. Defaults to {}
        """
        super().__init__()

        self._is_classification = True
        self.add_dependency(["from sklearn.linear_model import SGDClassifier ",
                             "from fedbiomed.common.training_plans import FedSGDClassifier"
                             ])

    def training_routine_hook(self) -> None:
        """Training routine of SGDClassifier.
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self._model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self) -> None:
        """Initialize the model parameter.
        """

        if 'verbose' not in self._model_args:
            self._model_args['verbose'] = 1
            self._params.update({'verbose': 1})

        self._verbose_capture_option = self._model_args["verbose"]

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

        for p in self._params:
            setattr(self._model, p, self._params[p])

    def get_learning_rate(self) -> List[float]:
        _lr_key = 'eta0'
        return super().get_learning_rate(_lr_key)

    def evaluate_loss(self, output: StringIO, epoch: int) -> float:
        """Evaluate the loss

        Args:
            output: output of the scikit-learn SGDClassifier model during training
            epoch: epoch number

        Returns:
            float: the value of the loss function in the case of binary classification, and the weighted average
                of the loss values for all classes in case of multiclass classification
        """
        _loss_collector = self._evaluate_loss_core(output, epoch)
        if not self._is_binary_classification:
            support = self._compute_support(self.target)
            loss = np.average(_loss_collector, weights=support)  # perform a weighted average
            logger.warning("Loss plot displayed on Tensorboard may be inaccurate (due to some plain" +
                           " SGD scikit learn limitations)")
        else:
            loss = _loss_collector[-1]
        return loss


class FedPerceptron(FedSGDClassifier):
    """Fed-BioMed federated wrapper of Perceptron class from scikit-learn. """

    def __init__(self):
        """Class constructor.
        """
        super().__init__()
        self._model = self._model_cls(loss='perceptron')
        self.add_dependency(["from fedbiomed.common.training_plans import FedPerceptron"
                             ])



# ############################################################################################3
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

        super().__init__(model_args)

        self.is_classification = True
        self.add_dependency([
            "from sklearn.naive_bayes import BernoulliNB"
        ])

    def training_routine_hook(self):
        """Training routine of BernoulliNB.
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self._model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self):
        """Initialize the model parameter.
        """

        if 'verbose' in self._model_args:
            logger.error("[TENSORBOARD ERROR]: cannot compute loss for BernoulliNB "
                         ": it needs to be implemented")

        for p in self._params:
            setattr(self._model, p, self._params[p])


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
        self.add_dependency([
            "from sklearn.naive_bayes  import GaussianNB"
        ])

    def training_routine_hook(self):
        """Training routine of GaussianNB.
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self._model.partial_fit(self.data, self.target, classes=classes)

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
        for p in self._params:
            setattr(self._model, p, self._params[p])


class FedMultinomialNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of FedMultinomialNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self, model_args):
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

    def __init__(self, model_args):
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

    def __init__(self, model_args):
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

    def __init__(self, model_args):
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

    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + \
            ": model FedMiniBatchDictionaryLearning not implemented yet "
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass
