"""TrainingPlan classes wrapping some types of scikit-learn models."""

from io import StringIO
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger

from ._sklearn_training_plan import SKLearnTrainingPlan


class FedSGDRegressor(SKLearnTrainingPlan):
    """Fed-BioMed training plan for scikit-learn SGDRegressor models."""

    _model_cls = SGDRegressor
    _model_dep = (
        "from sklearn.linear_model import SGDRegressor",
        "from fedbiomed.common.training_plans import FedSGDRegressor"
    )

    def __init__(self) -> None:
        """Initialize the sklearn SGDRegressor training plan."""
        super().__init__()
        self._is_regression = True

    def set_init_params(self) -> None:
        """Initialize the model's trainable parameters."""
        init_params = {
            'intercept_': np.array([0.]),
            'coef_': np.array([0.] * self._model_args['n_features'])
        }
        self._param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self._model, key, val)

    def training_routine_hook(self) -> None:
        """Training routine of SGDRegressor for an epoch."""
        # TODO:
        # * mutualize with SGDClassifier
        # * take `num_updates` into account
        # Gather start weights of the model and initialize zero gradients.
        param = {k: getattr(self._model, k) for k in self._param_list}
        grads = {k: np.zeros_like(v) for k, v in param.items()}
        # Iterate over data batches.
        for inputs, target in self.training_data_loader:
            # Iteratively accumulate sample-wise gradients, resetting weights.
            b_len = len(inputs.shape[0])
            for idx in range(b_len):
                self._model.partial_fit(inputs[idx:idx+1], target[idx])
                for key in self._param_list:
                    grads[key] += getattr(self._model, key)
                    setattr(self._model, key, param[key])
                    self._model.n_iter_ -= 1  # for non-constant learning rates
            # Compute the batch-averaged gradients and apply them.
            # Update the `param` values, and reset gradients to zero.
            for key in self._param_list:
                param[key] = grads[key] / b_len
                grads[key] = np.zeros_like(grads[key])
                setattr(self._model, key, param[key])
                self._model.n_iter_ += 1

    def parse_training_loss(
            self,
            output: StringIO,
            epoch: int
        ) -> float:
        """Parse the training loss from model training outputs."""
        losses = self._parse_training_losses(output, epoch)
        try:
            loss = losses[-1]
        except IndexError:
            logger.error(
                "Failed to parse any loss from captured outputs."
            )
            loss = float('nan')
        return loss


class FedSGDClassifier(SKLearnTrainingPlan):
    """Fed-BioMed training plan for scikit-learn SGDClassifier models."""

    _model_cls = SGDClassifier
    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedSGDClassifier"
    )

    def __init__(self) -> None:
        """Initialize the sklearn SGDClassifier training plan."""
        super().__init__()
        self._is_classification = True

    def set_init_params(self) -> None:
        """Initialize the model's trainable parameters."""
        # Set up zero-valued start weights, for binary of multiclass classif.
        n_classes = self._model_args["n_classes"]
        if n_classes == 2:
            init_params = {
                "intercept_": np.array((1,)),
                "coef_": np.zeros((1, self._model_args["n_features"]))
            }
        else:
            init_params = {
                "intercept_": np.zeros((n_classes,)),
                "coef_": np.zeros((n_classes, self._model_args["n_features"]))
            }
        # Assign these initialization parameters and retain their names.
        self._param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self._model, key, val)

    training_routine_hook = FedSGDRegressor.training_routine_hook

    def parse_training_loss(
            self,
            output: StringIO,
            epoch: int
        ) -> float:
        """Parse the training loss from model training outputs."""
        # TODO: raise or catch-and-log possible exceptions
        losses = self._parse_training_losses(output, epoch)
        if self._model_args["n_classes"] == 2:
            loss = losses[-1]
        else:
            support = self._compute_support(self.training_data_loader.get_target())
            loss = np.average(losses, weights=support)
            logger.warning(
                "Loss plot displayed on Tensorboard may be inaccurate "
                "(due to some scikit-learn SGDClassifier limitations)."
            )
        return loss


class FedPerceptron(FedSGDClassifier):
    """Fed-BioMed training plan for scikit-learn Perceptron models.

    This class inherits from FedSGDClassifier, and forces the wrapped
    scikit-learn SGDClassifier model to use a "perceptron" loss, that
    makes it equivalent to an actual scikit-learn Perceptron model.
    """

    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedPerceptron"
    )

    def __init__(self) -> None:
        """Class constructor."""
        super().__init__()
        self._model.set_params(loss="perceptron")

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
        ) -> None:
        model_args["loss"] = "perceptron"
        super().post_init(model_args, training_args)



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
        super().__init__()

        self.is_classification = True
        self.add_dependency([
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
        self.add_dependency([
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
