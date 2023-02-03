

from abc import abstractmethod
from io import StringIO
import sys
from typing import Any, Callable, Dict, List, Union, Iterator
from contextlib import contextmanager

import numpy as np
from declearn.model.sklearn import NumpyVector
from declearn.optimizer import Optimizer
from fedbiomed.common.exceptions import FedbiomedModelError
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier
import torch
from declearn.model.torch import TorchVector



class Model:
    
    model = None  #type = Union[nn.module, BaseEstimator]
    
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
        pass 
    @abstractmethod
    def update_weigths(self):
        pass
        

class TorchModel(Model):
    model =  None
    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiate the wrapper over a torch Module instance."""
        if not isinstance(model, torch.nn.Module):
            raise FedbiomedModelError(f"invalid argument for `model`: expecting a torch.nn.Module, but got {type(model)}")
        self.model = model

    def get_gradients(self, return_type: Callable = None) -> Any:
        """Return a TorchVector wrapping the gradients attached to the model."""
        super().get_gradients(return_type=return_type)
        gradients = {
            name: param.grad.detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients

    def get_weights(self, return_type: Callable = None) -> Any:
        """Return a TorchVector wrapping the model's parameters."""
        super().get_weights(return_type=return_type)
        parameters = {
            name: param.detach()
            for name, param in self.model.named_parameters()
        }
        if return_type is not None:
            parameters = return_type(parameters)
        return parameters

    def apply_updates(self, updates: TorchVector) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        with torch.no_grad():
            for name, update in updates.coefs.items():
                param = self.model.get_parameter(name)
                param.add_(update)
                
    def predict(self, inputs)-> np.ndarray:
        with torch.no_grad():
            pred = self.model(inputs) 
        return pred.numpy()

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
    def __init__(self, model):
        
        self._instance = Models[model.__name__](model())
        
     
    def __getattr__(self, item: str):

        """Wraps all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            AttributeError: If the attribute is not implemented

        """

        try:
            return self._instance.__getattribute__(item)
        except AttributeError:
            raise AttributeError(f"Error in SKlearnModel Builder: {item} not an attribute of {self._instance}")
        

class BaseSkLearnModel(Model):
    model = None
    default_lr_init: float = .1
    default_lr: str = 'constant'
    batch_size: int
    is_declearn_optim: bool
    param_list: List[str] = NotImplemented
    model_args: Dict[str, Any] = {}
    verbose: bool = NotImplemented
    grads: Dict[str, np.ndarray] = NotImplemented
    def __init__(
        self,
        model: BaseEstimator,

    ) -> None:
        """Instantiate the wrapper over a scikit-learn BaseEstimator."""
        if not isinstance(model, BaseEstimator):
            raise FedbiomedModelError(f"invalid argument for `model`: expecting an object extending from BaseEstimator, but got {model.__class__}")
        self.model = model
        # if len(param_list) == 0:
        #     raise FedbiomedModelError("Argument param_list can not be empty, but should contain model's layer names (as strings)")
        # self.param_list = param_list
        self.batch_size: int = 0
        self.is_declearn_optim = False  # TODO: to be changed when implementing declearn optimizers
        
        # if hasattr(model, "verbose"):
        #     self.verbose = True
        # else:
        #     self.verbose = False
        
    def init_training(self):
        self.param: Dict[str, np.ndarray] = {k: getattr(self.model, k) for k in self.param_list}
        self.grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.param.items()}
        
        if self.is_declearn_optim:
            self.set_learning_rate()
        

    def set_weights(self, weights: Dict[str, np.ndarray]):
        for key, val in weights.items():
                setattr(self.model, key, val)
        
    def get_weights(self, return_type: Callable = None) -> Any:
        
        """Return a NumpyVector wrapping the model's parameters."""
        super().get_weights(return_type=return_type)
        try:
            weights = {key: getattr(self.model, key) for key in self.param_list}
        except AttributeError as err:
            raise FedbiomedModelError(f"Unable to access weights of BaseEstimator model {self.model} (details {str(err)}")
        if return_type is not None:
            weights = return_type(weights)
        return weights

    def apply_updates(self, updates: Union[NumpyVector, Dict[str, np.ndarray]]) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        
        if isinstance(updates, dict):
            updates = NumpyVector(updates)
        for key, val in updates.coefs.items():
            setattr(self.model, key, val)
        self.model.n_iter_ += 1    
        self.batch_size = 0 
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs)
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, stdout: List[str] = None):
        if self.grads is NotImplemented:
            raise FedbiomedModelError("Training has not been instantiated: please run `init_training` method beforehand")
        self.batch_size += inputs.shape[0]
        with capture_stdout() as console:
            self.model.partial_fit(inputs, targets)
        if stdout is not None:
            stdout.append(console)
        for key in self.param_list:
            self.grads[key] += getattr(self.model, key)
            setattr(self.model, key, self.param[key])
        self.model.n_iter_ -= 1
    
    def get_gradients(self, return_type: Callable = None) -> Any:
        super().get_gradients(return_type=return_type)
        self.model.n_iter_ -= 1
        gradients: Dict[str, np.ndarray] = {}
        if self.is_declearn_optim:
            adjust = self.batch_size * self.get_learning_rate()[0]
            for key in self.param_list:
                gradients[key] = (self.get_weights() - self.grads[key]) / adjust
        else:
            for key in self.param_list:
                gradients[key] = self.grads[key] / self.batch_size
        self.model.n_iter_ += 1
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients
    
    
    def get_params(self, value: Any = None) -> Dict[str, Any]:
        if value is not None:
            return self.model.get_params(value)
        else: 
            return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    @abstractmethod
    def set_init_params(self):
        pass
    
    @abstractmethod
    def get_learning_rate(self) -> List[float]:
        pass
    
    @abstractmethod
    def set_learning_rate(self):
        pass


# TODO: check for `self.model.n_iter += 1` and `self.model.n_iter -= 1` if it makes sense
# TODO: agree on how to compute batch_size (needed for scaling): is the proposed method correct?

# ---- toolbox classes for getting learning rate and setting initial model parameters
class RegressorSkLearnModel(BaseSkLearnModel):
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
    _is_classification: bool = True
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
        

# --------- Models with appropriate methods ----- 
class SGDClassiferSKLearnModel(ClassifierSkLearnModel, SGDSkLearnModel):
    pass 

class MLPClassfierSKLearnModel(ClassifierSkLearnModel, MLPSklearnModel):
    pass

class SGDRegressorSKLearnModel(RegressorSkLearnModel, SGDSkLearnModel):
    pass


Models = {
    SGDClassifier.__name__: SGDClassiferSKLearnModel ,
    MLPClassifier.__name__: MLPClassfierSKLearnModel,
    SGDRegressor.__name__: SGDRegressorSKLearnModel
}