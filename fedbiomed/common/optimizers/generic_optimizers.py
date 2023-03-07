from abc import abstractmethod
from typing import Callable, Dict, List, Type, Union

from fedbiomed.common. models import Model, SkLearnModel
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.optimizer import Optimizer

import declearn
from declearn.model.api import Vector
import torch
import numpy as np


class GenericOptimizer:
    model: Model
    optimizer: Union[Optimizer, None]
    _step_method: Callable = NotImplemented
    _return_type: Union[None, Callable]
    def __init__(self, model: Model, optimizer: Union[Optimizer, None], return_type: Union[None, Callable] = None):
        self.model = model
        self.optimizer = optimizer
        self._return_type = return_type
        if isinstance(optimizer, declearn.optimizer.Optimizer):
            self._step_method = self.step_modules
        else:
            if hasattr(self,'step_native'):
                self._step_method = self.step_native
            else:
                raise FedbiomedOptimizerError(f"Optimizer {optimizer} has not `step_native` method, can not proceed")
        
            
    def step(self) -> Callable:
        if self._step_method is NotImplemented:
            raise FedbiomedOptimizerError("Error, method used for step not implemeted yet")
        return self._step_method()
    
    def step_modules(self):
        grad: Vector = self.model.get_gradients(self._return_type)
        weights: Vector = self.model.get_weights(self._return_type)
        updates = self.optimizer.step(grad, weights)
        self.model.apply_updates(updates)

    @classmethod
    def load_state(cls, state):
        # state: breakpoint content for optimizer
        return cls
    def save_state(self):
        pass
    def set_aux(self):
        pass
    def get_aux(self):
        pass
    def init_training(self):
        self.model.init_training()
        
    def send_model_to_device(self, device: torch.device):
        self.model.send_to_device(device)
        
    def zero_grad(self):
        # warning: specific for pytorch
        self.model.model.zero_grad()

    @abstractmethod
    def step_native(selfs):
        """_summary_

        Raises:
            FedbiomedOptimizerError: _description_
            FedbiomedOptimizerError: _description_
        """


class TorchOptimizer(GenericOptimizer):
    def __init__(self, model, optimizer: torch.optim.Optimizer, return_type=None):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise FedbiomedOptimizerError(f"Error, expected a `torch.optim` optimizer, but got {type(optimizer)}")
        super().__init__(model, optimizer, return_type=None)

    def setp_native(self):
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def get_learning_rate(self) -> List[float]:
        """Gets learning rate from value set in optimizer.

        !!! warning
            This function gathers the base learning rate applied to the model weights,
            including alterations due to any LR scheduler. However, it does not catch
            any adaptive component, e.g. due to RMSProp, Adam or such.

        Returns:
            List[float]: list of single learning rate or multiple learning rates
                (as many as the number of the layers contained in the model)
        """
        learning_rates = []
        params = self.optimizer.param_groups
        for param in params:
            learning_rates.append(param['lr'])
        return learning_rates

    def fed_prox(self, loss: torch.float, mu: Union[float, 'torch.float']) -> torch.float:
        loss += float(mu) / 2. * self.__norm_l2()
        return loss
    
    def scaffold(self):
        pass # FIXME: should we implement scaffold here?

    def __norm_l2(self) -> float:
        """Regularize L2 that is used by FedProx optimization

        Returns:
            L2 norm of model parameters (before local training)
        """
        norm = 0

        for current_model, init_model in zip(self.model.model().parameters(), self.model.init_params):
            norm += ((current_model - init_model) ** 2).sum()
        return norm
    
    
class SkLearnOptimizer(GenericOptimizer):
    
    def __init__(self, model: SkLearnModel, optimizer, return_type=None):
        if not isinstance(model, SkLearnModel):
            raise FedbiomedOptimizerError(f"Error in model argument: expected a `SkLearnModel` object, but got {type(model)}")
        super().__init__(model, optimizer, return_type=None)

    def step_native(self):
        gradients = self.model.get_gradients()
        lrate = self.model.get_learning_rate()[0]
        # revert back gradients to the batch averaged gradients
        gradients: Dict[str, np.ndarray] = {layer: val * lrate for layer, val in gradients.items()}
            
        self.model.apply_updates(gradients)

