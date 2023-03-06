from abc import abstractmethod
from typing import Callable

from fedbiomed.common import models
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.optimizer import Optimizer

import declearn


class GenericOptimizer:
    model: models
    optimizer: Optimizer
    _step_method: Callable = NotImplemented
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
        if isinstance(optimizer, declearn.optimizer.Optimizer):
            self._step_method = optimizer.step_modules
        else:
            self._step_method = optimizer.step_native
        
    def step(self, weights, gradients):
        if self._step_method is NotImplemented:
            raise FedbiomedOptimizerError("Error, method used for step not implemeted yet")
        return self._step_method(weights, gradients)
    
    def step_modules(self):
        pass
    
    def set_state(self):
        pass
    def get_state(self):
        pass
    @abstractmethod
    def step_native(self):
        pass

class TorchOptimizer(GenericOptimizer):
    def setp_native(self):
        pass
    
    
class SkLearnOptimizer(GenericOptimizer):
    def step_native(self):
        pass

