

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
import joblib
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Iterator
from contextlib import contextmanager

import numpy as np
from declearn.model.sklearn import NumpyVector
from declearn.optimizer import Optimizer
from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier
import torch
from declearn.model.torch import TorchVector



class Model(metaclass=ABCMeta):
    
    model : Union[torch.nn.Module, BaseEstimator]
    model_args: Dict[str, Any]
    
    @abstractmethod
    def init_training(self):
        """Initializes parameters before model training
        """
    @abstractmethod
    def train(self, inputs: Any, targets: Any, *args, **kwargs):
        """Trains model given inputs and targets data

        !!! warning "Pease run `init_training` method before running `train` method, so to initialize parameters needed for model training"

        !!! warning "This function may not update weights. You may need to call `apply_updates` to apply updates to the model"
        
        Args:
            inputs (Any): input (training) data.
            targets (Any): target values.
        """
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Returns model predictions given input values

        Args:
            inputs (Any): input values.

        Returns:
            Any: predictions.
        """

    @abstractmethod
    def apply_updates(self, updates: Any):
        """Applies updates to the model.

        Args:
            updates (Any): model updates.
        """
    @abstractmethod
    def get_weights(self, return_type: Callable = None) -> Any:
        """Returns weights of the model.

        Args:
            return_type (Callable, optional): Function that converts the dictionary mapping
            layers to model weights into another data structure. `return_type` should be used
            mainly with `declearn`'s `Vector`s. Defaults to None.

        Returns:
            Any: model's weights.
        """
    @abstractmethod
    def get_gradients(self, return_type: Callable = None) -> Any:
        """Returns computed gradients after training a model

        Args:
            return_type (Callable, optional): _description_. Defaults to None.
        """

    @abstractmethod
    def load(self, filename: str):
        """Loads model from a file.

        Args:
            path_file (str): path towards the file where the model has been saved.
        """
    @abstractmethod
    def save(self, filename: str):
        """Saves model into a file.

        Args:
            filename (str): path to the file, where will be savded the model.
        """
    @staticmethod
    def _validate_return_type(return_type: Optional[Callable] = None) -> None:
        """Checks that `return_type` argument is either a callble or None.
        Otherwise raises an error

        Args:
            return_type (Optional[Callable], optional): callable that will
            be used to convetrt a dictionry into another data structure (e.g. a declearn
            Vector). Defaults to None.

        Raises:
            FedbiomedModelError: raised if `return_type` argument is neither a callable nor `None`.
        """
        if not (return_type is None or callable(return_type)):
            raise FedbiomedModelError(f"argument return_type should be either None or callable, but got {type(return_type)} instead")



class TorchModel(Model):
    """PyTorch model wrapper that ease the handling of a pytorch model
    
    Attibutes:
    - model: torch.nn.Module. Pytorch model wrapped.
    - init_params: OrderedDict. Model initial parameters. Set when calling `init_training`
    """
    model: torch.nn.Module
    init_params: OrderedDict
    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiates the wrapper over a torch Module instance."""
        # if not isinstance(model, torch.nn.Module):
        #     raise FedbiomedModelError(f"invalid argument for `model`: expecting a torch.nn.Module, but got {type(model)}")
        self.model = model

    def get_gradients(self,
                      return_type: Callable[[Dict[str, torch.Tensor]], Any] = None) -> Union[Dict[str, torch.Tensor],Any]:
        """Returns a TorchVector wrapping the gradients attached to the model.
        
        Args:
            return_type (Callable, optional): callable that loads gradients into a 
            data structure and outputs gradients in this data structure. If not provided,
            returns gradient under a dictionary mapping model's layer names to theirs tensors. 
            Defaults to None. 
        
        Returns:
            Gradients in a dictionary mapping model's layer names to theirs tensors (if
            `return_type` argument is not provided) or in a data structure returned by `return_type`.
        """
        self._validate_return_type(return_type=return_type)
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if (param.requires_grad and param.grad is not None)
        }
        
        if len(gradients) < len(list(self.model.named_parameters())):
            logger.warning("Warning: can not retrieve all gradients from the model. Are you sure you have "
                           "trained the model beforehand?")
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients

    def get_weights(self,
                    only_trainable: bool = False,
                    return_type: Callable[[Dict[str, torch.Tensor]], Any] = None) -> Any:
        """Return a TorchVector wrapping the model's parameters.
        
        Args:
            only_trainable (bool, optional): whether to gather weights only on trainable layers (ie
                non frozen layers) or all layers (trainable and frozen). Defaults to False, (trainable and 
                frozen ones) 
            return_type (Callable, optional): callable that loads weights into a 
                data structure and outputs weights in this data structure. If not provided,
                returns weights under a dictionary mapping model's layer names to theirs tensors. 
                Defaults to None. 
        
        Returns:
            Model's weights in a dictionary mapping model's layer names to theirs tensors (I am going to change thatif
            `return_type` argument is not provided) or in a data structure returned by `return_type` Callable.
        """
        self._validate_return_type(return_type=return_type)
        parameters = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad or not only_trainable
        }
        if return_type is not None:
            parameters = return_type(parameters)
        return parameters

    def apply_updates(self, updates: Union[TorchVector, OrderedDict]) -> None:
        """Apply incoming updates to the wrapped model's parameters.
        
        Args:
            updates (Union[TorchVector, OrderedDict]): model updates to be added to the model.
        
        """
        iterator = self._get_iterator_model_params(updates)
        with torch.no_grad():
            for name, update in iterator:
                param = self.model.get_parameter(name)
                param.add_(update)
    
    def add_corrections_to_gradients(self, corrections: Union[TorchVector, Dict[str, torch.Tensor]]):
        """Adds values to attached gradients in the model"""
        iterator = self._get_iterator_model_params(corrections)

        for name, update in iterator:
            param = self.model.get_parameter(name)
            param.grad.add_(update.to(param.grad.device))

    def _get_iterator_model_params(self, model_params: Union[Dict[str, torch.Tensor], TorchVector]) -> Iterable[Tuple[str, torch.Tensor]]:
        """Returns an iterable from model_params, whether it is a 
        dictionary or a declearn's TorchVector

        Args:
            model_params (Union[Dict[str, torch.Tensor], TorchVector]): model parameters

        Raises:
            FedbiomedModelError: raised if argument `model_params` type is neither
            a TorchVector nor a dictionary

        Returns:
            Iterable[Tuple]: iterbale containing model parameters, that returns layer name and its value
        """
        if isinstance(model_params, TorchVector):
            
            iterator = model_params.coefs.items()
        elif isinstance(model_params, dict):
            iterator = model_params.items()
        else:
            raise FedbiomedModelError(f"Error, got a {type(model_params)} while expecting TorchVector or OrderedDict/Dict")
        return iterator

    def predict(self, inputs: torch.Tensor)-> np.ndarray:
        """Computes prediction given input data.

        Args:
            inputs (torch.Tensor): input data

        Returns:
            np.ndarray: model predictions returned as a numpy array
        """
        self.model.eval()  # pytorch switch for model inference-mode
        with torch.no_grad():
            pred = self.model(inputs) 
        return pred.cpu().numpy()
    
    def send_to_device(self, device: torch.device):
        """Sends model to device
        
        Args:
            device (torch.device): device set for using GPU or CPU.
        """
        return self.model.to(device)

    def init_training(self):
        """Initializes and sets attributes before model training.

        Initialized attributes:
        - init_params: copy of the initial parameters of the model
        
        """
        # initial aggregated model parameters
        self.init_params = deepcopy(list(self.model.parameters()))
        self.model.train()  # pytorch switch for training
        self.model.zero_grad()
        
    def train(self, inputs: torch.Tensor, targets: torch.Tensor,):
        # TODO: should we pass loss function here? and do the backward prop?
        if self.init_params is NotImplemented:
            raise FedbiomedModelError("Error, training has not been initialized, please initalized it beforehand")
        pass

    def load(self, filename: str) -> OrderedDict:
        # loads model from a file
        params = torch.load(filename)
        self.model.load_state_dict(params)
        return params
        
    def save(self, filename: str):
        torch.save(self.model.state_dict(), filename)

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
    Wrapper of Scikit learn model. It implements all abstract methods from 
    `Model`
    
    Attributes:
    - model: BaseEstimator. wrapped model
    - default_lr_init: float. Default value for setting learning rate to the scikit learn model. Needed
        for computing gradients. Set with `set_learning_rate` setter
    - default_lr: str. Default value for setting learning rate schedule to the scikit learn model. Needed for computing 
        gradients. Set with `set_learning_rate` setter
    - _batch_size: int. Internal counter that measures size of the batch.
    - _is_declearn_optim: bool. Switch that allows the use of declearn's optimizers
    - param_list: List[str]. List that contains layer attributes. Should be set when calling `set_init_params` method
    - updates: Dict[str, np.ndarray]. Contains model updates after a training. Set up when calling `init_training`
               method.
        
    Raises:
        FedbiomedModelError: raised if model is not as scikit learn `BaseEstimator` object

    """
    model: BaseEstimator
    default_lr_init: float = .1
    default_lr: str = 'constant'
    _batch_size: int
    _is_declearn_optim: bool
    param_list: List[str]
    _gradients: Dict[str, np.ndarray]
    updates: Dict[str, np.ndarray]  #replace `grads` from th poc
    def __init__(
        self,
        model: BaseEstimator,

    ) -> None:
        """Instantiate the wrapper over a scikit-learn BaseEstimator."""
        if not isinstance(model, BaseEstimator):
            err_msg = f"invalid argument for `model`: expecting an object extending from BaseEstimator, but got {model.__class__}"
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)
        self.model = model

        self._batch_size: int = 0
        self._is_declearn_optim: bool = False  # TODO: to be changed when implementing declearn optimizers
        self.param_list = NotImplemented
        self._gradients = NotImplemented
        self.updates = NotImplemented
        # FIXME: should we force model verbosity here?
        # if hasattr(model, "verbose"):
        #     self.verbose = True
        # else:
        #     self.verbose = False
        
    def init_training(self):
        """Initialises the training by setting up attributes.
        
        Attributes set:
         - param: initial parameters of the model
         - updates: attribute used to store model updates. Initially, it is arrays of zeros
         - _batch_size: internal counter that measure the batch_size, with respect to the data
            used for training model

        Raises:
            FedbiomedModelError: raised if `param_list` has not been defined
        """
        if self.param_list is NotImplemented:
            raise FedbiomedModelError("Attribute `param_list` is not defined: please define it beforehand")
        self.param: Dict[str, np.ndarray] = {k: getattr(self.model, k) for k in self.param_list}  # call it `param_init` so to be consistent with SklearnModel
        self.updates: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.param.items()}
        
        self._batch_size = 0 
        
        # if self.is_declearn_optim:
        #     self.disable_internal_optimizer()
        
    def _get_iterator_model_params(self, model_params: Union[Dict[str, np.ndarray], NumpyVector]) -> Iterable[Tuple[str, np.ndarray]]:
        """Returns an iterable from model_params, whether it is a dictionary or a `declearn`'s NumpyVector.

        Args:
            model_params (Union[Dict[str, np.ndarray], NumyVector]): model parameters

        Raises:
            FedbiomedModelError: raised if argument `model_params` type is neither
            a NumpyVector nor a dictionary

        Returns:
            Iterable[Tuple]: iterbale containing model parameters, that returns a mapping of model's layer names (actually model's
            name attributes corresponding to layer) and its value.
        """
        if isinstance(model_params, NumpyVector):
            return model_params.coefs.items()
        elif isinstance(model_params, dict):
            return model_params.items()
        else:
            raise FedbiomedModelError(f"Error, got a {type(model_params)} while expecting NumpyVector or OrderedDict/Dict")

    def set_weights(self, weights: Union[Dict[str, np.ndarray], NumpyVector]) -> BaseEstimator:
        """Sets model weights.

        Args:
            weights (Dict[str, np.ndarray]): model weights contained in a dictionary mapping layers names
            to its model paramaters (in numpy arrays)

        Returns:
            BaseEstimator: model wrapped updated with incoming weights
        """
        weights = self._get_iterator_model_params(weights)
        for key, val in weights:
            setattr(self.model, key, val.copy() if isinstance(val, np.ndarray) else val)
        return self.model

    def get_weights(self, return_type: Callable[[Dict[str, np.ndarray]], Any] = None) -> Any:
        """Returns model's parameters."""
        self._validate_return_type(return_type=return_type)
        weights = {}
        if self.param_list is NotImplemented:
            raise FedbiomedModelError("`param_list` not defined. You should have initialized the model beforehand (try calling `set_init_params`)")
        try:
             for key in self.param_list:
                val = getattr(self.model, key)
                weights[key] = val.copy() if isinstance(val, np.ndarray) else val

        except AttributeError as err:
            raise FedbiomedModelError(f"Unable to access weights of BaseEstimator model {self.model} (details {str(err)}")

        if return_type is not None:
            weights = return_type(weights)
        return weights

    def apply_updates(self, updates: Union[Dict[str, np.ndarray], NumpyVector]) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        updates = self._get_iterator_model_params(updates)
        
        for key, val in updates:
            w = getattr(self.model, key)
            setattr(self.model, key, val + w)
          

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Computes prediction given input data.

        Args:
            inputs (np.ndarray): input data

        Returns:
            np.ndarray: model predictions
        """
        return self.model.predict(inputs)
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, stdout: List[str] = None):
        """Trains scikit learn model and internally computes gradients 

        Args:
            inputs (np.ndarray): inputs data.
            targets (np.ndarray): targets, to be fit with inputs data
            stdout (List[str], optional): list of console outputs that have been collected
            during training, that contains losses values. Used to plot model losses. Defaults to None.

        Raises:
            FedbiomedModelError: raised if training has not been initialized
        """
        if self.updates is NotImplemented:
            raise FedbiomedModelError("Training has not been initialized: please run `init_training` method beforehand")
        self._batch_size: int = 0  # batch size counter
        
        # Iterate over the batch; accumulate sample-wise gradients (and loss).
        for idx in range(inputs.shape[0]):
            # Compute updated weights based on the sample. Capture loss prints.
            with capture_stdout() as console:
                self.model.partial_fit(inputs[idx:idx+1], targets[idx])
            if stdout is not None:
                stdout.append(console)
            for key in self.param_list:
                # Accumulate updated weights (weights + sum of gradients).
                # Reset the model's weights and iteration counter.
                self.updates[key] += getattr(self.model, key)
                setattr(self.model, key, self.param[key])  #resetting parameter to initial values
            
            self.model.n_iter_ -= 1
            self._batch_size += 1
        
        # compute gradients
        w = self.get_weights()
        self._gradients: Dict[str, np.ndarray] = {}
        if self._is_declearn_optim:
            adjust = self._batch_size * self.get_learning_rate()[0]
            
            for key in self.param_list:
                self._gradients[key] = ( w[key] * (1 - adjust) - self.updates[key]) / adjust
        else:
             # Compute the batch-averaged updated weights and apply them.
            for key in self.param_list:
                self._gradients[key] = self.updates[key] / self._batch_size - w[key]
        self.model.n_iter_ += 1
        
        # resetting updates
        self.updates: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.param.items()}

    def get_gradients(self, return_type: Callable[[Dict[str, np.ndarray]], Any] = None) -> Any:
        """Gets computed gradients

        Args:
            return_type (Callable, optional):  callable that loads gradients into a 
            data structure and outputs gradients in this data structure. If not provided,
            returns gradient under a dictionary mapping model's layer names to theirs tensors. 
            Defaults to None.

        Raises:
            FedbiomedModelError: raised if gradients have not been computed yet (ie model has not been trained)

        Returns:
            Any: Gradients in a dictionary mapping model's layer names to theirs tensors (if
            `return_type` argument is not provided) or in a data structure returned by `return_type`.
        """
        self._validate_return_type(return_type=return_type)
        if self._gradients is NotImplemented:
            raise FedbiomedModelError("Error, cannot get gradients if model hasnot been trained beforehand!")

        gradients: Dict[str, np.ndarray] = self._gradients
        
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients
    
    
    def get_params(self, value: Any = None) -> Dict[str, Any]:
        """Gets scikit learn model hyperparameters. Please refer to [`baseEstimator documentation`]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `get_params` method
        for further details

        Args:
            value (Any, optional): if specified, returns a specific hyperparameter, otherwise, returns a dictionary
            with all the hyperparameters. Defaults to None.

        Returns:
            Dict[str, Any]: dictionary mapping model hyperparameter names to their values
        """
        if value is not None:
            return self.model.get_params(value)
        else: 
            return self.model.get_params()

    def set_params(self, **params: Any) -> Dict[str, Any]:
        """Sets scikit learn model hyperparameters. Please refer to [`baseEstimator documentation`]
        [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html] `set_params` method
        for further details

        Args: 
            params (Any): new hyperparameters to set up the model.

        Returns:
            Dict[str, Any]: dictionary containing new hyperparameters values
        """
        self.model.set_params(**params)
        return params

    def load(self, filename: str):
        # FIXME: Security issue using pickles!
        with open(filename, "rb") as file:
            model = joblib.load(file)
        self.model = model
            
    def save(self, filename: str):
        with open(filename, "wb") as file:
            joblib.dump(self.model, file)

# ---- abstraction for sklearn models
    @abstractmethod
    def set_init_params(self, model_args: Dict, *args, **kwargs):
        """Zeroes scikit learn model parameters. Should be used before any training,
        as it sets the scikit learn model parameters and makes them accessible through the
        use of attributes.
        Model parameter attribute names will depend on the scikit learn model wrapped.
        
        Args: 
            model_args(Dict): dictionary that contains specifications for setting inital model
            parameters.
        """
    
    @abstractmethod
    def get_learning_rate(self) -> List[float]:
        """Retrieves learning rate of the model. Method implementation will
        depend on the attribute used to set up these arbitrary arguments
        
        Returns:
            List[float]: intial learning rate value(s); a single value if only on learning rate has been used, and
                     a list of several learning rates, one for each layer of the model.
        """
    
    @abstractmethod
    def disable_internal_optimizer(self):
        """Disables scikit learn internal optimizer by setting arbitrary learning rate
        parameters to the scikit learn model, in order to then compute its gradients.
        ''' warning "Call it only if using `declearn` optimizers"
        Method implementation will depend on the attribute used to set up these arbitrary arguments.
        """


# TODO: check for `self.model.n_iter += 1` and `self.model.n_iter -= 1` if it makes sense
# TODO: agree on how to compute batch_size (needed for scaling): is the proposed method correct?


class SGDSkLearnModel(BaseSkLearnModel):
    def get_learning_rate(self) -> List[float]:
        return [self.model.eta0]
    
    def disable_internal_optimizer(self):
        self.model.eta0 = self.default_lr_init
        self.model.learning_rate = self.default_lr
        self._is_declearn_optim = True

      
class MLPSklearnModel(BaseSkLearnModel):  # just for sake of demo
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
        self.param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self.model, key, val)


class SGDClassiferSKLearnModel(SGDSkLearnModel):
    """Toolbox class for Sklearn Classifier models based on SGD
    """
    _is_classification: bool = True
    _is_regression: bool = False
    #classes_: np.ndarray = NotImplemented
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
        self.param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self.model, key, val)
        # Also initialize the "classes_" slot with unique predictable labels.
        # FIXME: this assumes target values are integers in range(n_classes).
        setattr(self.model, "classes_", np.arange(n_classes))
        
 
class SkLearnModel():
    _instance: BaseSkLearnModel
    """Sklearn model builder. It wrapps one of Fed-BioMed `BaseSkLearnModel` object children, 
    by passing a [https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html]
    object to the constructor, as shown below.
    
    Usage:
    >>> from sklearn.linear_model import SGDClassifier
    >>> model = SkLearnModel(SGDClassifier)
    >>> model.set_weights(some_weights)  
    >>> type(model.model)
    <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>
    
    Attributes:
        _instance (BaseSkLearnModel): instance of BaseSkLearnModel
    """
    def __init__(self, model: Type[BaseEstimator]):
        """Constructor of the builder.

        Args:
            model (Type[BaseEstimator]): non-initalized `BaseEstimator` object
            
        Raises:
            FedBiomedModelError: raised if model does not belong to the implemented models.
            FedBiomedModelError: raised if `__name__` attribute does not belong to object. This may happen
                                when passing an instantiated object instead of the class object (eg instance of
                                SGDClassifier() instead of SGDClassifier object) 

        """
        if hasattr(model, '__name__'):
            try:
                self._instance = Models[model.__name__](model())
            except KeyError as ke:
                raise FedbiomedModelError(f"Error when building SkLearn Model: {model} has not been implemented in Fed-BioMed. Details: {ke}") from ke
        else:
            raise FedbiomedModelError(f"cannot build SkLearn Model: Model {model} don't have a `__name__` attribute. Are yousure you have not passed a"
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
            raise FedbiomedModelError(f"Error in SKlearnModel Builder: {item} not an attribute of {self._instance}")
    
    def __deepcopy__(self, memo) -> 'SkLearnModel':
        """
        Provides a deepcopy of the object. Copied object will have no shared references with the original model.
        !!!  warning "to be used with the `copy` built-in package of Python"
        
        Usage:
        >>> from sklearn.linear_model import SGDClassifier
        >>> model = SkLearnModel(SGDClassifier)
        >>> import copy
        >>> model_copy = copy.deepcopy(model)

        Args:
            memo (dict): dictionary fo completing new 

        Returns:
            SkLearnMode: deep copied object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

Models = {
    SGDClassifier.__name__: SGDClassiferSKLearnModel ,
    SGDRegressor.__name__: SGDRegressorSKLearnModel
}
