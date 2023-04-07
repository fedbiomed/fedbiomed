# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scikit-learn interfacing Model classes."""

import sys
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union, Iterator

import joblib
import numpy as np
from declearn.model.sklearn import NumpyVector
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.models import Model


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
        model_args: Dict storing additional parameters used to initialize the wrapped model.
        param_list: List that contains layer attributes. Should be set when calling `set_init_params` method

    Attributes: Class attributes:
        default_lr_init: Default value for setting learning rate to the scikit learn model. Needed
            for computing gradients. Set with `set_learning_rate` setter
        default_lr: Default value for setting learning rate schedule to the scikit learn model. Needed for computing
            gradients. Set with `set_learning_rate` setter
        is_classification: Boolean flag indicating whether the wrapped model is designed for classification
            or for regression supervised-learning tasks.
    """

    _model_type: ClassVar[Type[BaseEstimator]] = BaseEstimator
    # Instance attributes' annotations - merely for the docs parser.
    model: BaseEstimator
    model_args: Dict[str, Any]
    # Class attributes
    default_lr_init: ClassVar[float] = 0.1
    default_lr: ClassVar[str] = "constant"
    is_classification: ClassVar[bool]

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

    @staticmethod
    def _get_iterator_model_params(
        model_params: Union[Dict[str, np.ndarray], NumpyVector]
    ) -> Iterable[Tuple[str, np.ndarray]]:
        """Returns an iterable from model_params, whether it is a dictionary or a `declearn`'s NumpyVector.

        Args:
            model_params: model parameters or gradients

        Raises:
            FedbiomedModelError: raised if argument `model_params` type is neither
                a NumpyVector nor a dictionary

        Returns:
            Iterable containing model parameters, that returns a mapping of model's layer names
                (actually model's  name attributes corresponding to layer) and its value.
        """
        if isinstance(model_params, NumpyVector):
            return model_params.coefs.items()
        if isinstance(model_params, dict):
            return model_params.items()
        raise FedbiomedModelError(
            f"{ErrorNumbers.FB622.value} got a {type(model_params)} "
            "while expecting a NumpyVector or a dict"
        )

    def get_weights(
        self,
        as_vector: bool = False,
        only_trainable: bool = False,
    ) -> Union[Dict[str, np.ndarray], NumpyVector]:
        """Returns model's parameters, optionally as a declearn NumpyVector.

        Args:
            as_vector: Whether to wrap returned weights into a declearn Vector.
            only_trainable: Unused for scikit-learn models. (Whether to ignore
                non-trainable model parameters.)

        Raises:
            FedbiomedModelError: If the list of parameters are not defined.

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
                f"model {self.model} (details {err}"
            ) from err
        # Optionally encapsulate into a NumpyVector, else return as a dict.
        if as_vector:
            return NumpyVector(weights)
        return weights

    def set_weights(
        self,
        weights: Union[Dict[str, np.ndarray], NumpyVector],
    ) -> None:
        """Assign new values to the model's trainable weights.

        Args:
            weights: Model weights, as a dict mapping parameters' names to their
                numpy array, or as a declearn NumpyVector wrapping such a dict.
        """
        for key, val in self._get_iterator_model_params(weights):
            setattr(self.model, key, val.copy())

    def apply_updates(self, updates: Union[Dict[str, np.ndarray], NumpyVector]) -> None:
        """Apply incoming updates to the wrapped model's parameters.

        Args:
            updates: Model parameters' updates to add/apply existing model parameters.
        """
        for key, val in self._get_iterator_model_params(updates):
            wgt = getattr(self.model, key)
            setattr(self.model, key, wgt + val)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
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
        """Trains scikit learn model and internally computes gradients

        Args:
            inputs: inputs data.
            targets: targets, to be fit with inputs data
            stdout: list of console outputs that have been collected
                during training, that contains losses values. Used to plot model losses. Defaults to None.

        Raises:
            FedbiomedModelError: raised if training has not been initialized
        """
        batch_size = inputs.shape[0]
        w_init = self.get_weights(as_vector=False)  # type: Dict[str, np.ndarray]
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
                setattr(self.model, key, w_init[key])
            self.model.n_iter_ -= 1
        # Compute the batch-averaged, learning-rate-scaled gradients.
        # Note: w_init: {w_t}, w_updt: {w_t - eta_t * sum_{s=1}^B(grad_s)}
        #       hence eta_t * avg(grad_s) = w_init - (w_updt / B)
        self._gradients = {
            key: w_init[key] - (w_updt[key] / batch_size)
            for key in self.param_list
        }
        # Finally, increment the model's iteration counter.
        self.model.n_iter_ += 1

    def get_gradients(
        self,
        as_vector: bool = False,
    ) -> Union[Dict[str, np.ndarray], NumpyVector]:
        """Return computed gradients attached to the model.

        Args:
            as_vector: Whether to wrap returned gradients into a declearn Vector.

        Raises:
            FedbiomedModelError: If no gradients have been computed yet
                (i.e. the model has not been trained).

        Returns:
            Gradients, as a dictionary mapping parameters' names to their gradient's
                numpy array, or as a declearn NumpyVector wrapping such a dict.
        """
        if not self._gradients:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Cannot get gradients if model has not been trained beforehand!"
            )
        gradients = self._gradients
        if as_vector:
            return NumpyVector(gradients)
        return gradients

    def set_gradients(self, gradients: Union[Dict[str, np.ndarray], NumpyVector]) -> Dict[str, Any]:
        if isinstance(gradients, NumpyVector):
            gradients = gradients.coefs
        self._gradients = gradients
        return gradients

    def get_params(self, value: Any = None) -> Dict[str, Any]:
        """Gets scikit learn model hyperparameters.

        Please refer to [`baseEstimator documentation`]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `get_params` method
        for further details

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
        """Sets scikit learn model hyperparameters.

        Please refer to [BaseEstimator][sklearn.base.BaseEstimator]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `set_params` method
        for further details

        Args:
            params: new hyperparameters to set up the model.

        Returns:
            Dict[str, Any]: dictionary containing new hyperparameters values
        """
        self.model.set_params(**params)
        return params

    def check_changed_optimizer_params(self, init_model_args: Dict, to_string: bool = True) -> Tuple[bool, Union[List[str], str]]:
        new_params: Dict = self.get_params()
        changed_params: Union[List, str] = []
        for k, v in init_model_args.items():
            _param = new_params.get(k)
            if _param is not None and _param != v:
                changed_params.append(k)
        is_params_changed: bool = changed_params != []

        if to_string:
            changed_params = "\n".join(p + ",\n" for p in changed_params)
        return is_params_changed, changed_params

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

    @abstractmethod
    def disable_internal_optimizer(self) -> None:
        """Abstract method to apply;

        Disables scikit learn internal optimizer by setting arbitrary learning rate parameters to the
        scikit learn model, in order to then compute its gradients.

        ''' warning "Call it only if using `declearn` optimizers"
                Method implementation will depend on the attribute used to set up
                these arbitrary arguments.
        """
        # NOTE for developers:
        # subclasses should call `self._warn_overridden_optim_params`
        # on the first call to this function

    def _warn_overridden_optim_parameters(
        self, params: Collection[str]
    ) -> None:
        """Warn about non-default model parameters being overridden."""
        default = inspect.signature(type(self.model)).params
        changed = ", ".join(
            f"'{key}'"
            for key in params
            if self.model.get_parameter(key) != default[key].default
        )
        if changed:
            logger.warns(
                "The following non-default model parameters will be overridden"
                f" due to the disabling of the internal optimizer: {changed}."
            )


class SGDSkLearnModel(BaseSkLearnModel, metaclass=ABCMeta):
    """BaseSkLearnModel abstract subclass for geenric SGD-based models."""

    _model_type: ClassVar[Union[Type[SGDClassifier], Type[SGDRegressor]]]
    model: Union[SGDClassifier, SGDRegressor]  # merely for the docstring builder

    def get_learning_rate(self) -> List[float]:
        return [self.model.eta0]

    def disable_internal_optimizer(self) -> None:
        self.model.eta0 = self.default_lr_init
        self.model.learning_rate = self.default_lr


class SGDRegressorSKLearnModel(SGDSkLearnModel):
    """BaseSkLearnModel subclass for SGDRegressor models."""

    _model_type = SGDRegressor
    model: SGDRegressor  # merely for the docstring builder
    is_classification = False

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
    model: SGDClassifier  # merely for the docstring builder
    is_classification = True

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

    def get_learning_rate(self) -> List[float]:
        return [self.model.learning_rate_init]

    def disable_internal_optimizer(self) -> None:
        self.model.learning_rate_init = self.default_lr_init
        self.model.learning_rate = self.default_lr


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
