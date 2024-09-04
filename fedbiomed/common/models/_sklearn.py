# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scikit-learn interfacing Model classes."""

import sys
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Type, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.models import Model
from fedbiomed.common.logger import logger


@contextmanager
def capture_stdout() -> Iterator[List[str]]:
    """Context manager to capture console outputs (stdout).

    Returns:
        A list, empty at first, that will be populated with the line-wise
            strings composing the captured stdout upon exiting the context.
    """
    output = []  # type: List[str]
    stdout = sys.stdout
    str_io = StringIO()

    # Capture stdout outputs into the StringIO. Return yet-empty list.
    try:
        sys.stdout = str_io
        yield output
    # Restore sys.stdout, then parse captured outputs for loss values.
    finally:
        sys.stdout = stdout
        output.extend(str_io.getvalue().splitlines())


class BaseSkLearnModel(Model, metaclass=ABCMeta):
    """
    Wrapper of Scikit learn models.

    This class implements all abstract methods from the `Model` API, but adds some scikit-learn-specific ones
    that need implementing by its children.

    Attributes:
        model: Wrapped model
        param_list: List that contains layer attributes. Should be set when calling `set_init_params` method

    Attributes: Class attributes:
        is_classification: Boolean flag indicating whether the wrapped model is designed for classification
            or for regression supervised-learning tasks.
    """

    # Class attributes.
    is_classification: ClassVar[bool]
    _model_type: ClassVar[Type[BaseEstimator]] = BaseEstimator

    # Instance attributes' annotations - merely for the docs parser.
    model: BaseEstimator
    _null_optim_params: Dict[str, Any]
    _optim_params: Dict[str, Any] # optimizer parameters set by user

    def __init__(
        self,
        model: BaseEstimator,
    ) -> None:
        """Instantiate the wrapper over a scikit-learn BaseEstimator.

        Args:
            model: Model object as an instance of [BaseEstimator][sklearn.base.BaseEstimator]

        Raises:
            FedbiomedModelError: if model is not as scikit learn [BaseEstimator][sklearn.base.BaseEstimator] object
        """
        super().__init__(model)
        self._gradients: Dict[str, np.ndarray] = {}
        self.param_list: List[str] = []
        self._optim_params: Dict[str, Any] = {}

    def init_training(self):
        """Initialises the training by setting up attributes.

        Raises:
            FedbiomedModelError: raised if `param_list` has not been defined
        """
        if not self.param_list:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Attribute `param_list` is empty. You should "
                f"have initialized the model beforehand (try calling `set_init_params`)"
            )

    def get_weights(
        self,
        only_trainable: bool = False,
        exclude_buffers: bool = True
    ) -> Dict[str, np.ndarray]:
        """Return a copy of the model's trainable weights.

        Args:
            only_trainable: Unused for scikit-learn models. (Whether to ignore
                non-trainable model parameters.)
            exclude_buffers: Unused for scikit-learn models. (Whether to ignore
                buffers.)

        Raises:
            FedbiomedModelError: If the model parameters are not initialized.

        Returns:
            Model weights, as a dictionary mapping parameters' names to their
                numpy array, or as a declearn NumpyVector wrapping such a dict.
        """
        if not self.param_list:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Attribute `param_list` is empty. You should "
                f"have initialized the model beforehand (try calling `set_init_params`)"
            )
        # Gather copies of the model weights.
        weights = {}  # type: Dict[str, np.ndarray]
        try:
            for key in self.param_list:
                val = getattr(self.model, key)
                if not isinstance(val, np.ndarray):
                    raise FedbiomedModelError(
                        f"{ErrorNumbers.FB622.value}: SklearnModel parameter is not a numpy array."
                    )
                weights[key] = val.copy()
        except AttributeError as err:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Unable to access weights of BaseEstimator "
                f"model {self.model} (details {err})"
            ) from err
        return weights

    def flatten(self,
                only_trainable: bool = False,
                exclude_buffers: bool = True) -> List[float]:
        """Gets weights as flatten vector

        Args:
            only_trainable: Unused for scikit-learn models. (Whether to ignore
                non-trainable model parameters.)
            exclude_buffers: Unused for scikit-learn models. (Whether to ignore
                buffers.)

        Returns:
            to_list: Convert np.ndarray to a list if it is True.
        """

        weights = self.get_weights()
        flatten = []
        for _, w in weights.items():
            w_: List[float] = list(w.flatten().astype(float))
            flatten.extend(w_)

        return flatten

    def unflatten(
            self,
            weights_vector: List[float],
            only_trainable: bool = False,
            exclude_buffers: bool = True
    ) -> Dict[str, np.ndarray]:
        """Unflatten vectorized model weights

        Args:
            weights_vector: Vectorized model weights to convert dict
            only_trainable: Unused for scikit-learn models. (Whether to ignore
                non-trainable model parameters.)
            exclude_buffers: Unused for scikit-learn models. (Whether to ignore
                buffers.)

        Returns:
            Model dictionary
        """

        super().unflatten(weights_vector, only_trainable, exclude_buffers)

        weights_vector = np.array(weights_vector)
        weights = self.get_weights()
        pointer = 0

        params = {}
        for key, w in weights.items():
            num_param = w.size
            params[key] = weights_vector[pointer: pointer + num_param].reshape(w.shape)

            pointer += num_param

        return params

    def set_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> None:
        """Assign new values to the model's trainable weights.

        Args:
            weights: Model weights, as a dict mapping parameters' names
                to their numpy array.
        """
        self._assert_dict_inputs(weights)
        for key, val in weights.items():
            setattr(self.model, key, val.copy())

    def apply_updates(
        self,
        updates: Dict[str, np.ndarray],
    ) -> None:
        """Apply incoming updates to the wrapped model's parameters.

        Args:
            updates: Model parameters' updates to add (apply) to existing
                parameters' values.
        """
        self._assert_dict_inputs(updates)
        for key, val in updates.items():
            weights = getattr(self.model, key)
            setattr(self.model, key, weights + val)

    def predict(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Computes prediction given input data.

        Args:
            inputs: input data

        Returns:
            Model predictions
        """
        return self.model.predict(inputs)

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        stdout: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> None:
        """Run a training step, and record associated gradients.

        Args:
            inputs: inputs data.
            targets: targets, to be fit with inputs data.
            stdout: list of console outputs that have been collected
                during training, that contains losses values.
                Used to plot model losses. Defaults to None.

        Raises:
            FedbiomedModelError: if training has not been initialized.
        """
        batch_size = inputs.shape[0]
        w_init = self.get_weights()
        w_updt = {key: np.zeros_like(val) for key, val in w_init.items()}
        # Iterate over the batch; accumulate sample-wise gradients (and loss).
        for idx in range(batch_size):
            # Compute updated weights based on the sample. Capture loss prints.
            with capture_stdout() as console:
                self.model.partial_fit(inputs[idx : idx + 1], targets[idx])
            if stdout is not None:
                stdout.append(console)
            # Accumulate updated weights (weights + sum of gradients).
            # Reset the model's weights and iteration counter.
            for key in self.param_list:
                w_updt[key] += getattr(self.model, key)
                setattr(self.model, key, w_init[key].copy())
            self.model.n_iter_ -= 1
        # Compute the batch-averaged, learning-rate-scaled gradients.
        # Note: w_init: {w_t}, w_updt: {w_t - eta_t * sum_{s=1}^B(grad_s)}
        #       hence eta_t * avg(grad_s) = w_init - (w_updt / B)

        self._gradients = {
            key: w_init[key] - (w_updt[key] / batch_size)
            for key in self.param_list
        }

        # ------------------------------ WARNINGS ----------------------------------
        #
        # Warning 1: if `disable_internal_optimizer` has not been called before, gradients won't be scaled
        # (you will get un-scaled gradients, that need to be scaled back by dividing gradients by the learning rate)
        # here is a way to do so (with `lrate` as the learning rate):
        # ```python
        # for key, val in self._gradients.items():
        #        val /= lrate
        # ````
        # Warning 2:  `_gradients` has different meanings, when using `disable_internal_optimizer`
        # if it is not called (ie when using native sklearn optimizer), it is not plain gradients,
        # but rather the quantity `lr * grads`

        # Finally, increment the model's iteration counter.
        self.model.n_iter_ += 1
        # Nota: to restore sklearn internal optimizer, please call `enable_internal_optimizer`

    def get_gradients(
        self,
    ) -> Dict[str, np.ndarray]:
        """Return computed gradients attached to the model.

        Raises:
            FedbiomedModelError: If no gradients have been computed yet
                (i.e. the model has not been trained).

        Returns:
            Gradients, as a dict mapping parameters' names to their
                gradient's numpy array.
        """
        if not self._gradients:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Cannot get gradients if the "
                "model has not been trained beforehand."
            )
        gradients = self._gradients
        return gradients

    def set_gradients(self, gradients: Dict[str, np.ndarray]) -> None:
        # TODO: either document or remove this (useless) method
        self._gradients = gradients

    def get_params(self, value: Any = None) -> Dict[str, Any]:
        """Return the wrapped scikit-learn model's hyperparameters.

        Please refer to [`baseEstimator documentation`]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `get_params` method
        for further details.

        Args:
            value: if specified, returns a specific hyperparameter, otherwise, returns a dictionary
                with all the hyperparameters. Defaults to None.

        Returns:
            Dictionary mapping model hyperparameter names to their values
        """
        if value is not None:
            return self.model.get_params().get(value)
        return self.model.get_params()

    def set_params(self, **params: Any) -> Dict[str, Any]:
        """Assign some hyperparameters to the wrapped scikit-learn model.

        Please refer to [BaseEstimator][sklearn.base.BaseEstimator]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `set_params` method
        for further details.

        Args:
            params: new hyperparameters to assign to the model.

        Returns:
            Dict[str, Any]: dictionary containing new hyperparameter values.
        """
        self.model.set_params(**params)
        return params

    def disable_internal_optimizer(self) -> None:
        """Disable the scikit-learn internal optimizer.

        Calling this method alters the wrapped model so that raw gradients are
        computed and attached to it (rather than relying on scikit-learn to
        apply a learning rate that may be scheduled to vary along time).

        ''' warning "Call it only if using an external optimizer"
        """
        # Record initial params, then override optimizer ones.
        self._optim_params = self.get_params()
        self.set_params(**self._null_optim_params)
        # Warn about overridden values.
        changed_params: List[str] = []
        for key, val in self._null_optim_params.items():
            param = self._optim_params.get(key)
            if param is not None and param != val:
                changed_params.append(key)
        if changed_params:
            changed = ",\n\t".join(changed_params)
            logger.warning(
                "The following non-default model parameters were overridden "
                f"due to the disabling of the scikit-learn internal optimizer:\n\t{changed}",
                broadcast=True
            )

    def enable_internal_optimizer(self) -> None:
        """Enable the scikit-learn internal optimizer.

        Calling this method restores any model parameter previously overridden
        due to calling the counterpart `disable_internal_optimizer` method.
        """
        if self._optim_params:
            self.set_params(**self._optim_params)
            logger.debug("Internal Optimizer restored")

    def export(self, filename: str) -> None:
        """Export the wrapped model to a dump file.

        Args:
            filename: path to the file where the model will be saved.

        !!! info "Notes":
            This method is designed to save the model to a local dump
            file for easy re-use by the same user, possibly outside of
            Fed-BioMed. It is not designed to produce trustworthy data
            dumps and is not used to exchange models and their weights
            as part of the federated learning process.

        !!! warning "Warning":
            This method uses `joblib.dump`, which relies on pickle and
            is therefore hard to trust by third-party loading methods.
        """
        with open(filename, "wb") as file:
            joblib.dump(self.model, file)

    def reload(self, filename: str) -> None:
        """Import and replace the wrapped model from a dump file.

        Args:
            filename: path to the file where the model has been exported.

        !!! info "Notes":
            This method is designed to load the model from a local dump
            file, that might not be in a trustworthy format. It should
            therefore only be used to re-load data exported locally and
            not received from someone else, including other FL peers.

        Raises:
            FedbiomedModelError: if the reloaded instance is of unproper type.
        """
        model = self._reload(filename)
        if not isinstance(model, self._model_type):
            err_msg = (
                f"{ErrorNumbers.FB622.value}: unproper type for imported model"
                f": expected '{self._model_type}', but 'got {type(model)}'."
            )
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)
        self.model = model


    def _reload(self, filename: str) -> None:
        """Model-class-specific backend to the `reload` method.

        Args:
            filename: path to the file where the model has been exported.

        Returns:
            model: reloaded model instance to be wrapped, that will be type-
                checked as part of the calling `reload` method.
        """
        with open(filename, "rb") as file:
            model = joblib.load(file)
        return model

    # ---- abstraction for sklearn models
    @abstractmethod
    def set_init_params(self, model_args: Dict) -> None:
        """Zeroes scikit learn model parameters.

        Should be used before any training, as it sets the scikit learn model parameters
        and makes them accessible through the use of attributes. Model parameter attribute names
        will depend on the scikit learn model wrapped.

        Args:
            model_args: dictionary that contains specifications for setting initial model
        """

    @abstractmethod
    def get_learning_rate(self) -> List[float]:
        """Retrieves learning rate of the model. Method implementation will
        depend on the attribute used to set up these arbitrary arguments

        Returns:
            Initial learning rate value(s); a single value if only on learning rate has been used, and
                a list of several learning rates, one for each layer of the model.
        """


class SGDSkLearnModel(BaseSkLearnModel, metaclass=ABCMeta):
    """BaseSkLearnModel abstract subclass for geenric SGD-based models."""

    _model_type: ClassVar[Union[Type[SGDClassifier], Type[SGDRegressor]]]

    model: Union[SGDClassifier, SGDRegressor]  # merely for the docstring builder

    def __init__(self, model: BaseEstimator) -> None:
        super().__init__(model)
        self._null_optim_params: Dict[str, Any] = {
            'eta0': 1.0,
            'learning_rate': "constant",
        }
    def get_learning_rate(self) -> List[float]:
        return [self.model.eta0]


class SGDRegressorSKLearnModel(SGDSkLearnModel):
    """BaseSkLearnModel subclass for SGDRegressor models."""

    _model_type = SGDRegressor
    is_classification = False

    model: SGDRegressor  # merely for the docstring builder

    def set_init_params(self, model_args: Dict[str, Any]):
        """Initialize the model's trainable parameters."""
        init_params = {
            "intercept_": np.array([0.0]),
            "coef_": np.array([0.0] * model_args["n_features"]),
        }
        self.param_list = list(init_params)
        for key, val in init_params.items():
            setattr(self.model, key, val)


class SGDClassifierSKLearnModel(SGDSkLearnModel):
    """BaseSkLearnModel subclass for SGDClassifier models."""

    _model_type = SGDClassifier
    is_classification = True

    model: SGDClassifier  # merely for the docstring builder

    def set_init_params(self, model_args: Dict[str, Any]) -> None:
        """Initialize the model's trainable parameters."""
        # Set up zero-valued start weights, for binary of multiclass classif.
        n_classes = model_args["n_classes"]
        if n_classes == 2:
            init_params = {
                "intercept_": np.zeros((1,)),
                "coef_": np.zeros((1, model_args["n_features"])),
            }
        else:
            init_params = {
                "intercept_": np.zeros((n_classes,)),
                "coef_": np.zeros((n_classes, model_args["n_features"])),
            }
        # Assign these initialization parameters and retain their names.
        self.param_list = list(init_params)
        for key, val in init_params.items():
            setattr(self.model, key, val)
        # Also initialize the "classes_" slot with unique predictable labels.
        # FIXME: this assumes target values are integers in range(n_classes).
        setattr(self.model, "classes_", np.arange(n_classes))


class MLPSklearnModel(BaseSkLearnModel, metaclass=ABCMeta):  # just for sake of demo
    """BaseSklearnModel abstract subclass for multi-layer perceptron models."""

    _model_type: ClassVar[Union[Type[MLPClassifier], Type[MLPRegressor]]]
    model: Union[MLPClassifier, MLPRegressor]  # merely for the docstring builder

    def __init__(self, model: BaseEstimator) -> None:
        self._null_optim_params: Dict[str, Any] = {
            "learning_rate_init": 1.0,
            "learning_rate": "constant",
        }
        super().__init__(model)

    def get_learning_rate(self) -> List[float]:
        return [self.model.learning_rate_init]


SKLEARN_MODELS = {
    SGDClassifier.__name__: SGDClassifierSKLearnModel,
    SGDRegressor.__name__: SGDRegressorSKLearnModel,
}


class SkLearnModel:
    """Sklearn model builder.

    It wraps one of Fed-BioMed `BaseSkLearnModel` object children,
    by passing a (BaseEstimator)(https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
    object to the constructor, as shown below.

    **Usage**
    ```python
        from sklearn.linear_model import SGDClassifier
        model = SkLearnModel(SGDClassifier)
        model.set_weights(some_weights)
        type(model.model)
        # Output: <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>
    ```

    Attributes:
        _instance: instance of BaseSkLearnModel
    """

    def __init__(self, model: Type[BaseEstimator]):
        """Constructor of the model builder.

        Args:
            model: non-initialized [BaseEstimator][sklearn.base.BaseEstimator] object

        Raises:
            FedbiomedModelError: raised if model does not belong to the implemented models.
            FedbiomedModelError: raised if `__name__` attribute does not belong to object. This may happen
                when passing an instantiated object instead of the class object (e.g. instance of
                SGDClassifier() instead of SGDClassifier object)
        """
        if not isinstance(model, type):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}: 'SkLearnModel' received a '{type(model)}' instance as 'model' "
                "input while it was expecting a scikit-learn BaseEstimator subclass constructor."
            )
        if not issubclass(model, BaseEstimator):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}: 'SkLearnModel' received a 'model' class that is not "
                f"a scikit-learn BaseEstimator subclass: '{model}'."
            )
        if model.__name__ not in SKLEARN_MODELS:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}: 'SkLearnModel' received '{model}' as 'model' class, "
                f"support for which has not yet been implemented in Fed-BioMed."
            )
        self._instance: BaseSkLearnModel = SKLEARN_MODELS[model.__name__](model())

    def __getattr__(self, item: str):
        """Wraps all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            FedbiomedModelError: If the attribute is not implemented
        """
        try:
            return self._instance.__getattribute__(item)
        except AttributeError as exc:
            raise FedbiomedModelError(
                f"Error in SKlearnModel Builder: {item} not an attribute of {self._instance}"
            ) from exc

    def __deepcopy__(self, memo: Dict) -> "SkLearnModel":
        """Provides a deepcopy of the object.

        Copied object will have no shared references with the original model.

        !!!  warning "Warning"
            To be used with the `copy` built-in package of Python

        **Usage**
        ```python
            >>> from sklearn.linear_model import SGDClassifier
            >>> model = SkLearnModel(SGDClassifier)
            >>> import copy
            >>> model_copy = copy.deepcopy(model)
        ```

        Args:
            memo: dictionary for creating new object with new reference.

        Returns:
            Deep copied object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, val in self.__dict__.items():
            setattr(result, key, deepcopy(val, memo))
        return result
