# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import sys

from abc import abstractmethod, ABC
from copy import deepcopy
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, Iterator
from contextlib import contextmanager

import numpy as np
import joblib
from declearn.model.sklearn import NumpyVector
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor

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


class BaseSkLearnModel(Model):
    """
    Wrapper of Scikit learn model. It implements all abstract methods from `Model`

    Attributes:
        model: Wrapped model
        default_lr_init: Default value for setting learning rate to the scikit learn model. Needed
            for computing gradients. Set with `set_learning_rate` setter
        default_lr: Default value for setting learning rate schedule to the scikit learn model. Needed for computing
            gradients. Set with `set_learning_rate` setter
        _is_declearn_optim: Switch that allows the use of Declearn's optimizers
        param_list: List that contains layer attributes. Should be set when calling `set_init_params` method
    """

    _model_type = BaseEstimator
    default_lr_init: float = .1
    default_lr: str = 'constant'

    def __init__(
            self,
            model: BaseEstimator,
    ) -> None:
        """Instantiate the wrapper over a scikit-learn BaseEstimator.

        Args:
            model: Model object as an instance of [BaseEstimator][sklearn.base.BaseEstimator]

        Raises:
             FedbiomedModelError: raised if model is not as scikit learn [BaseEstimator][sklearn.base.BaseEstimator] object

        """
        super().__init__(model)
        self._is_declearn_optim: bool = False  # TODO: to be changed when implementing declearn optimizers
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
        # if self.is_declearn_optim:
        #     self.disable_internal_optimizer()

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
        elif isinstance(model_params, dict):
            return model_params.items()
        else:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value} got a {type(model_params)} "
                "while expecting a NumpyVector or a dict"
            )

    def set_weights(
            self,
            weights: Union[Dict[str, np.ndarray], NumpyVector],
    ) -> BaseEstimator:
        """Sets model weights.

        Args:
            weights: Model weights contained in a dictionary mapping layers names
                to its model parameters (in numpy arrays)

        Returns:
            Model wrapped updated with incoming weights
        """
        for key, val in self._get_iterator_model_params(weights):
            setattr(self.model, key, val.copy())
        return self.model

    def get_weights(
            self,
            as_vector: bool = False,
    ) -> Union[Dict[str, np.ndarray], NumpyVector]:
        """Returns model's parameters, optionally as a declearn NumpyVector.

        Args:
            as_vector: Whether to wrap returned weights into a declearn Vector.

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

    def apply_updates(
            self,
            updates: Union[Dict[str, np.ndarray], NumpyVector]
    ) -> None:
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
            stdout: Optional[List[str]] = None,
            **kwargs
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
                self.model.partial_fit(inputs[idx:idx + 1], targets[idx])
            if stdout is not None:
                stdout.append(console)
            # Accumulate updated weights (weights + sum of gradients).
            # Reset the model's weights and iteration counter.
            for key in self.param_list:
                w_updt[key] += getattr(self.model, key)
                setattr(self.model, key, w_init[key])
            self.model.n_iter_ -= 1
        # Compute the batch-averaged gradients (scaled by eta_t).
        # Note: w_init: {w_t}, w_updt: {w_t - eta_t sum_{s=1}^B(grad_s)}
        #       hence eta_t * avg(grad_s) = w_init - (w_updt / B)
        self._gradients = {
            key: w_init[key] - (w_updt[key] / batch_size)
            for key in self.param_list
        }
        # When using a declearn Optimizer, negate the learning rate.
        if self._is_declearn_optim:
            lrate = self.get_learning_rate()[0]
            for key, val in self._gradients.items():
                val /= lrate
        # Finally, increment the model's iteration counter.
        self.model.n_iter_ += 1

    def get_gradients(
            self,
            as_vector: bool = False,
    ) -> Union[Dict[str, np.ndarray], NumpyVector]:
        """Gets computed gradients

        Args:
            as_vector: Whether to wrap returned gradients into a declearn Vector.

        Raises:
            FedbiomedModelError: raised if gradients have not been computed yet (ie model has not been trained)

        Returns:
            Gradients, as a dictionary mapping parameters' names to their gradient's
                numpy array, or as a declearn NumpyVector wrapping such a dict.
         """
        if self._gradients is None:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Cannot get gradients if model has not been trained beforehand!"
            )
        gradients = self._gradients
        if as_vector:
            return NumpyVector(gradients)
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
            return self.model.get_params(value)
        else:
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

    def load(self, filename: str):
        """Loads model from a file.

        Args:
            filename: path towards the file where the model has been saved.
        """
        # FIXME: Security issue using pickles!
        with open(filename, "rb") as file:
            model = joblib.load(file)
        self.model = model

    def save(self, filename: str):
        """Saves model into a file.

        Args:
            filename: path to the file, where will be saved the model.
        """
        with open(filename, "wb") as file:
            joblib.dump(self.model, file)

    # ---- abstraction for sklearn models
    @abstractmethod
    def set_init_params(self, model_args: Dict, *args, **kwargs):
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
    def disable_internal_optimizer(self):
        """Abstract method to apply;

        Disables scikit learn internal optimizer by setting arbitrary learning rate parameters to the
        scikit learn model, in order to then compute its gradients.

        ''' warning "Call it only if using `declearn` optimizers"
                Method implementation will depend on the attribute used to set up
                these arbitrary arguments.
        """


# TODO: check for `self.model.n_iter += 1` and `self.model.n_iter -= 1` if it makes sense
# TODO: agree on how to compute batch_size (needed for scaling): is the proposed method correct?


class SGDSkLearnModel(BaseSkLearnModel, ABC):
    def get_learning_rate(self) -> List[float]:
        return [self.model.eta0]

    def disable_internal_optimizer(self):
        self.model.eta0 = self.default_lr_init
        self.model.learning_rate = self.default_lr
        self._is_declearn_optim = True


class MLPSklearnModel(BaseSkLearnModel, ABC):  # just for sake of demo
    def get_learning_rate(self) -> List[float]:
        return [self.model.learning_rate_init]

    def disable_internal_optimizer(self):
        self.model.learning_rate_init = self.default_lr_init
        self.model.learning_rate = self.default_lr
        self._is_declearn_optim = True


class SGDRegressorSKLearnModel(SGDSkLearnModel):
    """Toolbox class for Sklearn Regression models bsed on SGD
    """
    _is_classification: bool = False
    _is_regression: bool = True

    def set_init_params(self, model_args: Dict[str, Any]):
        """Initialize the model's trainable parameters."""
        init_params = {
            'intercept_': np.array([0.]),
            'coef_': np.array([0.] * model_args['n_features'])
        }
        self.param_list = list(init_params)
        for key, val in init_params.items():
            setattr(self.model, key, val)


class SGDClassifierSKLearnModel(SGDSkLearnModel):
    """Toolbox class for Sklearn Classifier models based on SGD
    """
    _is_classification: bool = True
    _is_regression: bool = False

    def set_init_params(self, model_args: Dict[str, Any]) -> None:
        """Initialize the model's trainable parameters."""
        # Set up zero-valued start weights, for binary of multiclass classif.
        n_classes = model_args["n_classes"]
        if n_classes == 2:
            init_params = {
                "intercept_": np.zeros((1,)),
                "coef_": np.zeros((1, model_args["n_features"]))
            }
        else:
            init_params = {
                "intercept_": np.zeros((n_classes,)),
                "coef_": np.zeros((n_classes, model_args["n_features"]))
            }
        # Assign these initialization parameters and retain their names.
        self.param_list = list(init_params)
        for key, val in init_params.items():
            setattr(self.model, key, val)
        # Also initialize the "classes_" slot with unique predictable labels.
        # FIXME: this assumes target values are integers in range(n_classes).
        setattr(self.model, "classes_", np.arange(n_classes))


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
    _instance: BaseSkLearnModel

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
        if hasattr(model, '__name__'):
            try:
                self._instance = Models[model.__name__](model())
            except KeyError as ke:
                raise FedbiomedModelError(
                    f"{ErrorNumbers.FB622.value}. Error when building SkLearn Model: {model} "
                    f"has not been implemented in Fed-BioMed. Details: {ke}") from ke
        else:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Cannot build SkLearn Model: Model {model} don't "
                f"have a `__name__` attribute. Are you sure you have not passed a"
                " sklearn object instance instead of the object class")

    def __getattr__(self, item: str):
        """Wraps all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            FedbiomedModelError: If the attribute is not implemented
        """

        try:
            return self._instance.__getattribute__(item)
        except AttributeError:
            raise FedbiomedModelError(
                f"Error in SKlearnModel Builder: {item} not an attribute of {self._instance}"
            )

    def __deepcopy__(self, memo: Dict) -> 'SkLearnModel':
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
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result


Models = {
    SGDClassifier.__name__: SGDClassifierSKLearnModel,
    SGDRegressor.__name__: SGDRegressorSKLearnModel
}
