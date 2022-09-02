"""
Fed-BioMed encapsulation of SKLearn base classes.
"""


from io import StringIO

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger

from ._sklearn_training_plan import SKLearnTrainingPlan


class FedPerceptron(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of Perceptron class from scikit-learn.

    Attributes:
        model: an instance of the sklearn class that we are wrapping (static attribute).
    """

    model = Perceptron()

    def __init__(self, model_args: dict = None):
        """Class constructor.

        Args:
            model_args: model arguments. Defaults to {}
        """
        if model_args is None:
            model_args = {}

        super().__init__(model_args)

        if "verbose" not in model_args:
            self.model_args["verbose"] = 1
            self.params.update({"verbose": 1})

        self._is_classification = True

        self._verbose_capture_option = self.model_args["verbose"]

        # Instantiate the model
        self.set_init_params()
        self.add_dependency(["from sklearn.linear_model import Perceptron "])

    def training_routine_hook(self) -> None:
        """Training hook of Perceptron.

        Initializes the data loader and calls the `partial_fit` method.
        """
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True
        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self) -> None:
        """Initialize the model parameters."""
        self.param_list = ["intercept_", "coef_"]
        init_params = {
            "intercept_": np.array([0.0])
            if (self.model_args["n_classes"] == 2)
            else np.array([0.0] * self.model_args["n_classes"]),
            "coef_": np.array([0.0] * self.model_args["n_features"]).reshape(
                1, self.model_args["n_features"]
            )
            if (self.model_args["n_classes"] == 2)
            else np.array(
                [0.0] * self.model_args["n_classes"] * self.model_args["n_features"]
            ).reshape(self.model_args["n_classes"], self.model_args["n_features"]),
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

    def evaluate_loss(self, output: StringIO, epoch: int) -> float:
        """Evaluate the loss.

        Args:
            output: output of the scikit-learn perceptron model during training
            epoch: epoch number

        Returns:
            float: the value of the loss function in the case of binary classification, and the weighted average
                of the loss values for all classes in case of multiclass classification
        """
        _loss_collector = self._evaluate_loss_core(output, epoch)
        if not self._is_binary_classification:
            support = self._compute_support(self.target)
            loss = np.average(
                _loss_collector, weights=support
            )  # perform a weighted average
            logger.warning(
                "Loss plot displayed on Tensorboard may be inaccurate (due to some plain"
                + " SGD scikit learn limitations)"
            )
        else:
            loss = _loss_collector[-1]
        return loss


class FedSGDRegressor(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of SGDRegressor class from scikit-learn.

    Attributes:
        model: an instance of the sklearn class that we are wrapping (static attribute).
    """

    model = SGDRegressor()

    def __init__(self, model_args: dict = None):
        """
        Sklearn SGDRegressor model.

        Args:
            model_args: model arguments. Defaults to {}
        """
        if model_args is None:
            model_args = {}
        super().__init__(model_args)

        if "verbose" not in model_args:
            self.model_args["verbose"] = 1
            self.params.update({"verbose": 1})

        # specific for SGDRegressor
        self._is_regression = True
        self._verbose_capture_option = self.model_args["verbose"]

        # Instantiate the model
        self.set_init_params()

        self.add_dependency(["from sklearn.linear_model import SGDRegressor "])

    def training_routine_hook(self) -> None:
        """
        Training routine of SGDRegressor.
        """
        (self.data, self.target) = self.training_data_loader
        self.model.partial_fit(self.data, self.target)

    def set_init_params(self) -> None:
        """
        Initialize the model parameter.
        """
        self.param_list = ["intercept_", "coef_"]
        init_params = {
            "intercept_": np.array([0.0]),
            "coef_": np.array([0.0] * self.model_args["n_features"]),
        }
        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

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
        model: an instance of the sklearn class that we are wrapping (static attribute).
    """

    model = SGDClassifier()

    def __init__(self, model_args: dict = None):
        """
        Sklearn SGDClassifier model.

        Args:
            model_args: model arguments. Defaults to {}
        """
        if model_args is None:
            model_args = {}
        super().__init__(model_args)

        # if verbose is not provided in model_args set it to true and add it to self.params
        if "verbose" not in model_args:
            self.model_args["verbose"] = 1
            self.params.update({"verbose": 1})

        self._is_classification = True
        self._verbose_capture_option = self.model_args["verbose"]

        # Instantiate the model
        self.set_init_params()

        self.add_dependency(["from sklearn.linear_model import SGDClassifier "])

    def training_routine_hook(self) -> None:
        """Training routine of SGDClassifier."""
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self) -> None:
        """Initialize the model parameter."""
        self.param_list = ["intercept_", "coef_"]
        init_params = {
            "intercept_": np.array([0.0])
            if (self.model_args["n_classes"] == 2)
            else np.array([0.0] * self.model_args["n_classes"]),
            "coef_": np.array([0.0] * self.model_args["n_features"]).reshape(
                1, self.model_args["n_features"]
            )
            if (self.model_args["n_classes"] == 2)
            else np.array(
                [0.0] * self.model_args["n_classes"] * self.model_args["n_features"]
            ).reshape(self.model_args["n_classes"], self.model_args["n_features"]),
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])

        for p in self.params:
            setattr(self.model, p, self.params[p])

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
            loss = np.average(
                _loss_collector, weights=support
            )  # perform a weighted average
            logger.warning(
                "Loss plot displayed on Tensorboard may be inaccurate (due to some plain"
                + " SGD scikit learn limitations)"
            )
        else:
            loss = _loss_collector[-1]
        return loss


# ############################################################################################3
class FedBernoulliNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of BernoulliNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    model = BernoulliNB()

    def __init__(self, model_args: dict = {}):
        """Sklearn BernoulliNB model.

        Args:
            model_args: model arguments. Defaults to {}
        """

        msg = ErrorNumbers.FB605.value + " FedBernoulliNB not implemented."
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

        super().__init__(model_args)

        self.is_classification = True
        if "verbose" in model_args:
            logger.error(
                "[TENSORBOARD ERROR]: cannot compute loss for BernoulliNB "
                ": it needs to be implemented"
            )

        self.set_init_params()

        self.add_dependency(["from sklearn.naive_bayes import BernoulliNB"])

    def training_routine_hook(self):
        """Training routine of BernoulliNB."""
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self):
        """Initialize the model parameter."""
        for p in self.params:
            setattr(self.model, p, self.params[p])


class FedGaussianNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of FedGaussianNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    model = GaussianNB()

    def __init__(self, model_args: dict = {}):
        """
        Sklearn GaussianNB model.

        Args:
            model_args: model arguments. Defaults to {}
        """

        msg = ErrorNumbers.FB605.value + " FedGaussianNB not implemented."
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

        super().__init__(model_args)
        self.is_classification = True

        if "verbose" in model_args:
            logger.error(
                "[TENSORBOARD ERROR]: cannot compute loss for GaussianNB "
                ": it needs to be implemeted"
            )

        self.set_init_params()

        self.add_dependency(["from sklearn.naive_bayes  import GaussianNB"])

    def training_routine_hook(self):
        """Training routine of GaussianNB."""
        (self.data, self.target) = self.training_data_loader
        classes = self._classes_from_concatenated_train_test()
        if classes.shape[0] < 3:
            self._is_binary_classification = True

        self.model.partial_fit(self.data, self.target, classes=classes)

    def set_init_params(self):
        """Initialize the model parameter."""

        self.param_list = ["intercept_", "coef_"]
        init_params = {
            "intercept_": np.array([0.0])
            if (self.model_args["n_classes"] == 2)
            else np.array([0.0] * self.model_args["n_classes"]),
            "coef_": np.array([0.0] * self.model_args["n_features"]).reshape(
                1, self.model_args["n_features"]
            )
            if (self.model_args["n_classes"] == 2)
            else np.array(
                [0.0] * self.model_args["n_classes"] * self.model_args["n_features"]
            ).reshape(self.model_args["n_classes"], self.model_args["n_features"]),
        }

        for p in self.param_list:
            setattr(self.model, p, init_params[p])
        for p in self.params:
            setattr(self.model, p, self.params[p])


class FedMultinomialNB(SKLearnTrainingPlan):
    """Fed-BioMed federated wrapper of FedMultinomialNB class from scikit-learn.

    !!! info "Not implemented yet!"
        This class has not yet been implemented.
    """

    def __init__(self, model_args):
        msg = ErrorNumbers.FB605.value + " FedMultinomialNB not implemented."
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
        msg = (
            ErrorNumbers.FB605.value
            + ": model FedPassiveAggressiveClassifier not implemented yet "
        )
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
        msg = (
            ErrorNumbers.FB605.value
            + ": model FedPassiveAggressiveRegressor not implemented yet "
        )
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
        msg = (
            ErrorNumbers.FB605.value + ": model FedMiniBatchKMeans not implemented yet "
        )
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
        msg = (
            ErrorNumbers.FB605.value
            + ": model FedMiniBatchDictionaryLearning not implemented yet "
        )
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine_hook(self):
        pass
