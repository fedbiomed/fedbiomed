from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Type, Union
from fedbiomed.common.constants import TrainingPlans

from fedbiomed.common. models import Model, SkLearnModel
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.logger import logger
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer


import declearn
from declearn.model.api import Vector
from declearn.model.torch import TorchVector
from declearn.model.sklearn import NumpyVector
import torch
import numpy as np


class GenericOptimizer:
    model: Model
    optimizer: Union[FedOptimizer, None]
    _step_method: Callable = NotImplemented
    _return_type: Union[None, Callable]
    def __init__(self, model: Model, optimizer: Union[FedOptimizer, None], return_type: Union[None, Callable] = None):
        self.model = model
        self.optimizer = optimizer
        self._return_type = return_type
        # if isinstance(optimizer, declearn.optimizer.Optimizer):
        #     self._step_method = self.step_modules
        # else:
        #     if hasattr(self,'step_native'):
        #         self._step_method = self.step_native
        #     else:
        #         raise FedbiomedOptimizerError(f"Optimizer {optimizer} has not `step_native` method, can not proceed")
        
            
    # def step(self) -> Callable:
    #     logger.debug("calling steps")
    #     if self._step_method is NotImplemented:
    #         raise FedbiomedOptimizerError("Error, method used for step not implemeted yet")
    #     #self._step_method()
    #     if isinstance(self.optimizer, declearn.optimizer.Optimizer):
    #         self.step_modules()
    #     else:
    #         self.step_native()
    
    def step_modules(self):
        logger.debug("calling step_modules: return type"+str(type(self._return_type)))
        grad: Vector = self.model.get_gradients(self._return_type)
        weights: Vector = self.model.get_weights(return_type=self._return_type)
        updates = self.optimizer.step(grad, weights)
        self.model.apply_updates(updates)
        print("MODEL", self.model.model.state_dict())

    @classmethod
    def build(cls, tp_type: TrainingPlans, model: Model, optimizer: Optional[Union[torch.optim.Optimizer, FedOptimizer]]=None) -> 'BaseOptimizer':
        if tp_type == TrainingPlans.TorchTrainingPlan:
            if isinstance(optimizer, (FedOptimizer)):
                # TODO: add return types
                return TorchOptimizer(model, optimizer, return_type=TorchVector)
            elif isinstance(optimizer, torch.optim.Optimizer):
                return NativeTorchOptimizer(model, optimizer)
            else:
                raise FedbiomedOptimizerError(f"Can not build optimizer from {optimizer}")
        elif tp_type == TrainingPlans.SkLearnTrainingPlan:
            if isinstance(optimizer, (FedOptimizer)):
                return SkLearnOptimizer(model, optimizer, return_type=NumpyVector)
            elif optimizer is None:
                return NativeSkLearnOptimizer(model, optimizer)
            else:
                raise FedbiomedOptimizerError(f"Can not build optimizer from {optimizer}")
        else:
            
            raise FedbiomedOptimizerError(f"Training Plan {tp_type} unknown")
            
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

    def train_model(self, inputs, target, stdout: Optional[List] = None):
        self.model.train(inputs, target, stdout)

    @abstractmethod
    def step(self):
        """_summary_
        """
    @abstractmethod
    def step_native(self):
        """_summary_

        Raises:
            FedbiomedOptimizerError: _description_
            FedbiomedOptimizerError: _description_
        """
    @abstractmethod
    def get_learning_rate(self)-> List[float]:
        """_summary_

        Raises:
            FedbiomedOptimizerError: _description_
            FedbiomedOptimizerError: _description_
            FedbiomedOptimizerError: _description_

        Returns:
            _type_: _description_
        """

# class GenericOptimizer(BaseOptimizer):
#     def __init__(self):
#         pass
    


       
class TorchOptimizer(GenericOptimizer):
    def zero_grad(self):
        # warning: specific for pytorch
        try:
            self.model.model.zero_grad()
        except AttributeError as err:
            raise FedbiomedOptimizerError(f"Model has no method named `zero_grad`: are you sure you are using a PyTorch TrainingPlan?. Details {err}") from err
    def send_model_to_device(self, device: torch.device):
        # for pytorch specifically
        self.model.send_to_device(device)

    def step(self):
        self.step_modules()

class NativeTorchOptimizer(GenericOptimizer):
    def __init__(self, model, optimizer: torch.optim.Optimizer, return_type=None):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise FedbiomedOptimizerError(f"Error, expected a `torch.optim` optimizer, but got {type(optimizer)}")
        super().__init__(model, optimizer, return_type=None)

    def setp(self):
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def get_learning_rate(self) -> List[float]:
        """Gets learning rate from value set in Pytorch optimizer.

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

    def send_model_to_device(self, device: torch.device):
        # for pytorch specifically
        self.model.send_to_device(device)

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
    def __init__(self, model: Model, optimizer: Union[FedOptimizer, None], return_type: Union[None, Callable] = None):
        super().__init__(model, optimizer, return_type)
        # self.model.disable_internal_optimizer()
        # is_param_changed, param_changed = self.model.check_changed_optimizer_params()
        # if is_param_changed:
        #     msg = "The following parameter(s) has(ve) been detected in the model_args but will be disabled when using a declearn Optimizer: please specify those values in the training_args or in the init_optimizer method"
        #     msg += param_changed
        #     logger.warning(msg)
    def get_learning_rate(self) -> List[float]:
        return self.model.get_learning_rate()
    
    def step(self):
        self.step_modules()
        
class NativeSkLearnOptimizer(GenericOptimizer):
    
    def __init__(self, model: SkLearnModel, optimizer=None, return_type=None):
        if not isinstance(model, SkLearnModel):
            raise FedbiomedOptimizerError(f"Error in model argument: expected a `SkLearnModel` object, but got {type(model)}")
        super().__init__(model, optimizer, return_type=None)

    def step(self):
        gradients = self.model.get_gradients()
        lrate = self.model.get_learning_rate()[0]
        # revert back gradients to the batch averaged gradients
        gradients: Dict[str, np.ndarray] = {layer: val * lrate for layer, val in gradients.items()}
            
        self.model.apply_updates(gradients)


