

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Union

import numpy as np
from declearn.model.sklearn import NumpyVector
from declearn.optimizer import Optimizer
from fedbiomed.common.exceptions import FedbiomedModelError
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
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
    
    
class SkLearnModel(Model):
    model = None
    default_lr_init: float = .1
    default_lr: str = 'constant'
    batch_size: int
    _is_declearn_optim: bool
    def __init__(
        self,
        model: BaseEstimator,
        param_list: List[str],
        is_declearn_optim: bool
    ) -> None:
        """Instantiate the wrapper over a scikit-learn BaseEstimator."""
        if not isinstance(model, BaseEstimator):
            raise FedbiomedModelError(f"invalid argument for `model`: expecting a BaseEstimator, but got {type(model)}")
        self.model = model
        if len(param_list) == 0:
            raise FedbiomedModelError("Argument param_list can not be empty, but should contain model's layer names (as strings)")
        self.param_list = param_list
        self.batch_size: int = 0
        
        self.param: Dict[str, np.ndarray] = {k: getattr(self._model, k) for k in self.param_list}
        self.grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.param.items()}
        
        self._is_declearn_optim = is_declearn_optim
        if is_declearn_optim:
            self._set_raw_lrate()
        

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

    def apply_updates(self, updates: NumpyVector) -> None:
        """Apply incoming updates to the wrapped model's parameters."""
        self.model.n_iter_ -= 1
        for key, val in updates.coefs.items():
            setattr(self.model, key, val)
        self.model.n_iter_ += 1    
        self.batch_size = 0 
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs)
    
    def train(self, inputs: np.ndarray, targets: np.ndarray):
        self.batch_size += inputs.shape[0]
        self.model.partial_fit(inputs, targets)
        for key in self.param_list:
            self.grads[key] += getattr(self.model, key)
            setattr(self.model, key, self.param[key])
        
    
    def get_gradients(self, return_type: Callable = None) -> Any:
        super().get_weights(return_type=return_type)
        self.model.n_iter_ -= 1
        gradients: Dict[str, np.ndarray] = {}
        if self._is_declearn_optim:
            adjust = self.batch_size * self._get_raw_lrate()
            for key in self.param_list:
                gradients[key] = (self.get_weights() - self.grads[key]) / adjust
        else:
            for key in self.param_list:
                gradients[key] = self.grads[key] / self.batch_size
        self.model.n_iter_ += 1
        if return_type is not None:
            gradients = return_type(gradients)
        return gradients
        
    def set_init_params(self):
        # for multi classifiers
        pass
    
    @abstractmethod
    def _get_raw_lrate(self):
        pass
    
    @abstractmethod
    def _set_raw_lrate(self):
        pass

class SGDSkLearnModel(SkLearnModel):
    def _get_raw_lrate(self):
        return self.model.eta0
    
    def _set_raw_lrate(self):
        self.model.eta0 = self.default_lr_init
        self.model.learning_rate = self.default_lr


class MLPSklearnModel(SkLearnModel):  # just for sake of demo
    def _get_raw_lrate(self):
        return self.model.learning_rate_init
    
    def _set_raw_lrate(self):
        self.model.learning_rate_init = self.default_lr_init
        self.model.learning_rate = self.default_lr