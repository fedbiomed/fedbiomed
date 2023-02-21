

from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
import joblib
import sys
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Iterator
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



class Model:
    
    model = None  #type = Union[nn.module, BaseEstimator]
    model_args: Dict[str, Any] = {}
    
    @abstractmethod
    def init_training(self):
        pass
    @abstractmethod
    def train(self, inputs: Any, targets: Any, loss_func: Callable = None) -> Any:
        pass
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        pass
    @abstractmethod
    def load(self, path_file:str):
        pass
    
    @abstractmethod
    def set_weights(self, weights: Any):
        pass

    def get_weights(self, return_type: Callable = None):
        if not (return_type is None or callable(return_type)):
            raise FedbiomedModelError(f"argument return_type should be either None or callable, but got {type(return_type)} instead")
        
    def get_gradients(self, return_type: Callable = None):
        if not (return_type is None or callable(return_type)):
            raise FedbiomedModelError(f"argument return_type should be either None or callable, but got {type(return_type)} instead")


    @abstractmethod
    def update_weigths(self):
        pass
    @abstractmethod
    def load(self, filenmae:str):
        pass
        
    @abstractmethod
    def save(self, filename:str):
        pass


class TorchModel(Model):
    """PyTorch model wrapper that ease the handling of a pytorch model
    
    Attibutes:
    - model: torch.nn.Module. Pytorch model wrapped.
    - init_params: OrderedDict. Model initial parameters. Set when calling `init_training`
    """
    model: torch.nn.Module =  None
    init_params: OrderedDict
    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiate the wrapper over a torch Module instance."""
        # if not isinstance(model, torch.nn.Module):
        #     raise FedbiomedModelError(f"invalid argument for `model`: expecting a torch.nn.Module, but got {type(model)}")
        self.model = model
        # initial aggregated model parameters
        #self.init_params = deepcopy(list(self.model().parameters()))

    def get_gradients(self, return_type: Callable[[Dict[str, torch.Tensor]], Any] = None) -> Union[Dict[str, torch.Tensor],Any]:
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
        super().get_gradients(return_type=return_type)
        gradients = {
            name: param.grad.detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is not None
        }
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients

    def get_weights(self, return_type: Callable[[Dict[str, torch.Tensor]], Any] = None) -> Any:
        """Return a TorchVector wrapping the model's parameters.
        
        Args:
            return_type (Callable, optional): callable that loads weights into a 
            data structure and outputs weights in this data structure. If not provided,
            returns weights under a dictionary mapping model's layer names to theirs tensors. 
            Defaults to None. 
        
        Returns:
            Model's weights in a dictionary mapping model's layer names to theirs tensors (if
            `return_type` argument is not provided) or in a data structure returned by `return_type` Callable.
        """
        super().get_weights(return_type=return_type)
        parameters = {
            name: param.detach()
            for name, param in self.model.named_parameters()
        }
        if return_type is not None:
            parameters = return_type(parameters)
        return parameters

    def apply_updates(self, updates: Union[TorchVector, OrderedDict]) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        iterator = self._get_iterator_model_params(updates)
        with torch.no_grad():
            for name, update in iterator:
                param = self.model.get_parameter(name)
                param.add_(update)
    
    def apply_gradients(self, gradients: Union[TorchVector, Dict[str, torch.Tensor]]):
        """Adds values to attached gradients in the model"""
        iterator = self._get_iterator_model_params(gradients)

        for name, update in iterator:
            param = self.model.get_parameter(name)
            param.grad.add_(update.to(param.grad.device))

    def _get_iterator_model_params(self, model_params) -> Iterable[Tuple[str, torch.Tensor]]:
        """Returns an iterable from model_params, whether it is a 
        dictionary or a declearn's TorchVector

        Args:
            model_params (_type_): model parameters

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
        return pred.numpy()
    
    def send_to_device(self, device: torch.device):
        """sends model to device"""
        return self.model.to(device)
    

    def init_training(self):
        """Initializes and Sets attributes before model training.
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

class SkLearnModel():
    def __init__(self, model: BaseEstimator):
        self._instance = Models[model.__name__](model())
        
     
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
    # @classmethod
    # def load(cls, filename: str):
    #     with open(filename, "rb") as file:
    #         model = joblib.load(file)
    #     return cls(model)   

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
    - batch_size: int. Internal counter that measures size of the batch.
    - is_declearn_optim: bool. Switch that allows the use of declearn's optimizers
    - param_list: List[str]. List that contains layer attributes. Should be set when calling `set_init_params` method
    - updates: Dict[str, np.ndarray]. Contains model updates after a training. Set up when calling `init_training`
               method.
        
    Raises:
        FedbiomedModelError: raised if model is not as scikit learn `BaseEstimator` object

    """
    model = None
    default_lr_init: float = .1
    default_lr: str = 'constant'
    batch_size: int
    is_declearn_optim: bool
    param_list: List[str] = NotImplemented
    _gradients: Dict[str, np.ndarray] = NotImplemented
    #model_args: Dict[str, Any] = {}
    updates: Dict[str, np.ndarray] = NotImplemented  #replace `grads` from th poc
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
        # if len(param_list) == 0:
        #     raise FedbiomedModelError("Argument param_list can not be empty, but should contain model's layer names (as strings)")
        # self.param_list = param_list
        self.batch_size: int = 0
        self.is_declearn_optim = False  # TODO: to be changed when implementing declearn optimizers
        
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
         - batch_size: internal counter that measure the batch_size, with respect to the data
            used for training model

        Raises:
            FedbiomedModelError: raised if `param_list` has not been defined
        """
        if self.param_list is NotImplemented:
            raise FedbiomedModelError("Attribute `param_list` is not defined: please define it beforehand")
        self.param: Dict[str, np.ndarray] = {k: getattr(self.model, k) for k in self.param_list}  # call it `param_init` so to be consistent with SklearnModel
        self.updates: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.param.items()}
        
        self.batch_size = 0 
        
        if self.is_declearn_optim:
            # set a constant learning rate, that will be used to extract gradient
            self.set_learning_rate()
        

    def set_weights(self, weights: Dict[str, np.ndarray]) -> BaseEstimator:
        """Sets model weights

        Args:
            weights (Dict[str, np.ndarray]): model weights contained in a dictionary mapping layers names
            to its model paramaters (in numpy arrays)

        Returns:
            BaseEstimator: model wrapped updated with incoming weights
        """
        for key, val in weights.items():
            setattr(self.model, key, val)
        return self.model

    def get_weights(self, return_type: Callable = None) -> Any:
        """Returns model's parameters."""
        super().get_weights(return_type=return_type)
        if self.param_list is NotImplemented:
            raise FedbiomedModelError("`param_list` not defined. You should have initialized the model beforehand (try calling `set_init_params`)")
        try:
            weights = {key: getattr(self.model, key) for key in self.param_list}
        except AttributeError as err:
            raise FedbiomedModelError(f"Unable to access weights of BaseEstimator model {self.model} (details {str(err)}")
        if return_type is not None:
            weights = return_type(weights)
        return weights

    def apply_updates(self, updates: Dict[str, np.ndarray]) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        
        w = self.get_weights()
        for key, val in updates.items():
            setattr(self.model, key, val + w[key])
        self.model.n_iter_ += 1    
        
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
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
        self.batch_size += inputs.shape[0]
        with capture_stdout() as console:
            self.model.partial_fit(inputs, targets)
        if stdout is not None:
            stdout.append(console)
        for key in self.param_list:
            # cumul grad
            self.updates[key] += getattr(self.model, key)
            setattr(self.model, key, self.param[key])  #resetting parameter to initial values
        
        self.model.n_iter_ -= 1
        
        # compute gradients
        w = self.get_weights()
        self._gradients: Dict[str, np.ndarray] = {}
        if self.is_declearn_optim:
            adjust = self.batch_size * self.get_learning_rate()[0]
            
            for key in self.param_list:
                self._gradients[key] = ( w[key] * (1 - adjust) - self.updates[key]) / adjust
        else:
            for key in self.param_list:
                self._gradients[key] = self.updates[key] / self.batch_size - w[key]
    
    def get_gradients(self, return_type: Callable = None) -> Any:
        """Gets computed gradients

        Args:
            return_type (Callable, optional): _description_. Defaults to None.

        Raises:
            FedbiomedModelError: _description_

        Returns:
            Any: _description_
        """
        super().get_gradients(return_type=return_type)
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
            Dict[str, Any]: dictionary mapping hyperparameter names to their values
        """
        if value is not None:
            return self.model.get_params(value)
        else: 
            return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    def load(self, filename: str):
        with open(filename, "rb") as file:
            model = joblib.load(file)
        self.model = model
            
    def save(self, filename: str):
        with open(filename, "wb") as file:
            joblib.dump(self.model, file)

# ---- abstraction for sklearn models
    @abstractmethod
    def set_init_params(self):
        """Zeroes scikit learn model parameters. Should be used before any training,
        as it sets the scikit learn model parameters and makes them accessible.
        Mode parameters will depend on the scikit learn model
        """
        pass
    
    @abstractmethod
    def get_learning_rate(self) -> List[float]:
        """Retrieves learning rate of the model. Method implementation will
        depend on the attribute used to set up these arbitrary arguments"""
        pass
    
    @abstractmethod
    def set_learning_rate(self):
        """Sets arbitrary learning rate parameters to the scikit learn model, 
        in order to then compute its gradients. Method implementation will
        depend on the attribute used to set up these arbitrary arguments
        """
        pass
    
# TODO: check for `self.model.n_iter += 1` and `self.model.n_iter -= 1` if it makes sense
# TODO: agree on how to compute batch_size (needed for scaling): is the proposed method correct?

# ---- toolbox classes for getting learning rate and setting initial model parameters
# ---- they implement either:
# ---- + `set_init_params` method to initialise model weights to 0 and adds appriopriate attributes pointing to the model weigths
# -----   and the `param_list` attribute
# ---- + `get_learning_rate` and `set_learning_rate` methods that enable setting / retrieving learning rate related parameters


class RegressorSkLearnModel(BaseSkLearnModel):
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


class ClassifierSkLearnModel(BaseSkLearnModel):
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
        

class SGDSkLearnModel(BaseSkLearnModel):
    def get_learning_rate(self) -> List[float]:
        return [self.model.eta0]
    
    def set_learning_rate(self):
        self.model.eta0 = self.default_lr_init
        self.model.learning_rate = self.default_lr

class MLPSklearnModel(BaseSkLearnModel):  # just for sake of demo
    def get_learning_rate(self) -> List[float]:
        return [self.model.learning_rate_init]
    
    def set_learning_rate(self):
        self.model.learning_rate_init = self.default_lr_init
        self.model.learning_rate = self.default_lr
        

# --------- Model objects with appropriate methods ----- 
class SGDClassiferSKLearnModel(ClassifierSkLearnModel, SGDSkLearnModel):
    pass 

class SGDRegressorSKLearnModel(RegressorSkLearnModel, SGDSkLearnModel):
    pass


Models = {
    SGDClassifier.__name__: SGDClassiferSKLearnModel ,
    SGDRegressor.__name__: SGDRegressorSKLearnModel
}