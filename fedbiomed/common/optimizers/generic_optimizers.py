from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
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



class BaseOptimizer(metaclass=ABCMeta):
    model: Model
    optimizer: Union[FedOptimizer, None]

    def __init__(self, model: Model, optimizer: Union[FedOptimizer, None]):
        if not isinstance(model, (Model, SkLearnModel)):
            raise FedbiomedOptimizerError(f"Expected an instance of fedbiomed.common.model.Model or fedbiomed.common.model.SkLearnModel but got {model}")
        self.model = model
        self.optimizer = optimizer

    def init_training(self):
        self.model.init_training()

    def train_model(self, inputs, target, stdout: Optional[List] = None):
        self.model.train(inputs, target, stdout)

    @abstractmethod
    def step(self):
        """_summary_
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
    

class BaseDeclearnOptimizer(BaseOptimizer):
    def __init__(self, model: Model, optimizer: Union[FedOptimizer, declearn.optimizer.Optimizer, None]):
        logger.debug("Using declearn optimizer")
        if isinstance(optimizer, declearn.optimizer.Optimizer):
            optimizer = FedOptimizer.from_declearn_optimizer(optimizer)
        super().__init__(model, optimizer)
        self.optimizer.init_round()

    def step_modules(self):

        grad: Vector = self.model.get_gradients(as_vector=True)
        weights: Vector = self.model.get_weights(as_vector=True)
        updates = self.optimizer.step(grad, weights)
        self.model.apply_updates(updates)

    def step(self):
        self.step_modules()
    def get_learning_rate(self) -> List[float]:
        return self.model.get_learning_rate()
    
    def set_aux(self, aux: Dict[str, Any]):
        self.optimizer.process_aux_var(aux)
        
    def get_aux(self) -> Optional[Dict[str, Any]]:
        aux = self.optimizer.collect_aux_var()
        return aux

    @classmethod
    def load_state(cls, model, optim_state: Dict):
        # state: breakpoint content for optimizer
        relaoded_optim = FedOptimizer.load_state(optim_state)
        return cls(model, relaoded_optim)

    def save_state(self) -> Dict:
        # TODO: implement this method for loading state on Researcher side
        optim_state = self.optimizer.get_state()
        return optim_state


class DeclearnTorchOptimizer(BaseDeclearnOptimizer):
    def zero_grad(self):
        # warning: specific for pytorch
        try:
            self.model.model.zero_grad()
        except AttributeError as err:
            raise FedbiomedOptimizerError(f"Model has no method named `zero_grad`: are you sure you are using a PyTorch TrainingPlan?. Details {err}") from err

    def send_model_to_device(self, device: torch.device):
        # for pytorch specifically
        self.model.send_to_device(device)

    def get_learning_rate(self) -> List[float]:
        return [self.optimizer._optimizer.lrate]

class DeclearnSklearnOptimizer(BaseDeclearnOptimizer):
    
    def step(self):
        # convert batch averaged gradients into gradients efore conputation
        lrate = self.model.get_learning_rate()[0]
        if int(lrate) != 0:
            
            gradients = self.model.get_gradients()
            self.model.gradients = {layer: val / lrate for layer, val in gradients.items()}
        else:
            # Nota: if learning rate equals 0, there will be no updates applied during SGD
            logger.warning("Learning rate set to 0: no gradient descent will be performed!")   
        super().step()

    def optimizer_post_processing(self, model_args: Dict):
        self.model.disable_internal_optimizer()
        is_param_changed, param_changed = self.model.check_changed_optimizer_params(model_args)
        if is_param_changed:
            msg = "The following parameter(s) has(ve) been detected in the model_args but will be disabled when using a declearn Optimizer: please specify those values in the training_args or in the init_optimizer method"
            msg += "\nParameters changed:\n"
            msg += param_changed
            logger.warning(msg)    

class NativeTorchOptimizer(BaseOptimizer):
    def __init__(self, model, optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise FedbiomedOptimizerError(f"Error, expected a `torch.optim` optimizer, but got {type(optimizer)}")
        super().__init__(model, optimizer)
        logger.debug("using native torch optimizer")

    def step(self):
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
    

# class DeclearnSkLearnOptimizer(BaseOptimizer):
#     def __init__(self, model: Model, optimizer: Union[FedOptimizer, None], return_type: Union[None, Callable] = None):
#         super().__init__(model, optimizer, return_type)
#         # self.model.disable_internal_optimizer()
#         # is_param_changed, param_changed = self.model.check_changed_optimizer_params()
#         # if is_param_changed:
#         #     msg = "The following parameter(s) has(ve) been detected in the model_args but will be disabled when using a declearn Optimizer: please specify those values in the training_args or in the init_optimizer method"
#         #     msg += param_changed
#         #     logger.warning(msg)

    
#     def step(self):
#         self.step_modules()
        
class NativeSkLearnOptimizer(BaseOptimizer):
    
    def __init__(self, model: SkLearnModel, optimizer=None):
        if not isinstance(model, SkLearnModel):
            raise FedbiomedOptimizerError(f"Error in model argument: expected a `SkLearnModel` object, but got {type(model)}")
        super().__init__(model, optimizer)
        logger.debug("Using native Sklearn Optimizer")

    def step(self):
        gradients = self.model.get_gradients()
        #lrate = self.model.get_learning_rate()[0]
        # revert back gradients to the batch averaged gradients
        #gradients: Dict[str, np.ndarray] = {layer: val * lrate for layer, val in gradients.items()}

        self.model.apply_updates(gradients)

    def optimizer_post_processing(self, model_args: Dict):
        pass
    
    def get_learning_rate(self) -> List[float]:
        return self.model.get_learning_rate()

TORCH_OPTIMIZERS = {
    FedOptimizer: DeclearnTorchOptimizer,
    torch.optim.Optimizer: NativeTorchOptimizer
}

SKLEARN_OPTIMIZERS = {
    FedOptimizer: DeclearnSklearnOptimizer,
    None: NativeSkLearnOptimizer
}
TRAININGPLAN_OPTIMIZERS = {
    TrainingPlans.TorchTrainingPlan: TORCH_OPTIMIZERS,
    TrainingPlans.SkLearnTrainingPlan: SKLEARN_OPTIMIZERS,
}


class OptimizerBuilder:
    TORCH_OPTIMIZERS = {
    FedOptimizer: DeclearnTorchOptimizer,
    torch.optim.Optimizer: NativeTorchOptimizer
}

    SKLEARN_OPTIMIZERS = {
        FedOptimizer: DeclearnSklearnOptimizer,
        None: NativeSkLearnOptimizer
    }

    def __init__(self) -> None:    
         
        self.builder = {
            TrainingPlans.TorchTrainingPlan: self.build_torch,
            TrainingPlans.SkLearnTrainingPlan: self.build_sklearn
            }
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
    


    # def set_optimizers_for_training_plan(self, tp_type: TrainingPlans):
    #     self._training_plan_type: TrainingPlans = tp_type
    #     try:
    #         self._optimizers_available: Dict = TRAININGPLAN_OPTIMIZERS[tp_type]
    #     except KeyError:
    #         raise FedbiomedOptimizerError(f"Unknown TrainingPlan: {tp_type}")
    
    @staticmethod
    def build_torch(tp_type, model, optimizer):
        try:
            optimizer_wrapper: BaseOptimizer = OptimizerBuilder.TORCH_OPTIMIZERS[OptimizerBuilder.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"Optimizer {optimizer} is not compatible with training plan {tp_type}"
            
            raise FedbiomedOptimizerError(err_msg)
        return optimizer_wrapper(model, optimizer)
    
    @staticmethod
    def build_sklearn(tp_type, model, optimizer):
        try:
            optimizer_wrapper: BaseOptimizer = OptimizerBuilder.SKLEARN_OPTIMIZERS[OptimizerBuilder.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"Optimizer {optimizer} is not compatible with training plan {tp_type}" + \
            "\nHint: If If you want to use only native scikit learn optimizer, please do not define a `init_optimizer` method in the TrainingPlan"
            
            raise FedbiomedOptimizerError(err_msg)
        return optimizer_wrapper(model, optimizer)

    def build(self, tp_type: TrainingPlans, model: Model, optimizer: Optional[Union[torch.optim.Optimizer, FedOptimizer]]=None) -> 'BaseOptimizer':

        # if self._optimizers_available is None:
        #     raise FedbiomedOptimizerError("error, no training_plan set, please run `set_optimizers_for_training_plan` beforehand")
        try:
            return self.builder[tp_type](tp_type, model, optimizer)
            #optimizer_wrapper: BaseOptimizer = self._optimizers_available[self.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"Unknown Training Plan type {tp_type} "
            
            raise FedbiomedOptimizerError(err_msg)
        #return optimizer_wrapper(model, optimizer)
        # if tp_type == TrainingPlans.TorchTrainingPlan:
        #     if isinstance(optimizer, (FedOptimizer)):
        #         # TODO: add return types
        #         return DeclearnTorchOptimizer(model, optimizer, return_type=TorchVector)
        #     elif isinstance(optimizer, torch.optim.Optimizer):
        #         return NativeTorchOptimizer(model, optimizer)
        #     else:
        #         raise FedbiomedOptimizerError(f"Can not build optimizer from {optimizer}")
        # elif tp_type == TrainingPlans.SkLearnTrainingPlan:
        #     if isinstance(optimizer, (FedOptimizer)):
        #         return DeclearnSkLearnOptimizer(model, optimizer, return_type=NumpyVector)
        #     elif optimizer is None:
        #         return NativeSkLearnOptimizer(model, optimizer)
        #     else:
        #         raise FedbiomedOptimizerError(f"Can not build optimizer from {optimizer}")
        # else:
            
        #     raise FedbiomedOptimizerError(f"Unknown Training Plan type {tp_type} ")

    @staticmethod
    def get_parent_class(optimizer: Union[None, Any]) -> Union[None, Type]:
        if optimizer is None:
            return None
        if hasattr(type(optimizer), '__bases__'):
            if type(optimizer).__bases__[0]  is object:
                # in this case, `optimizer` is already the parent class (it only has `object`as parent class)
                return type(optimizer)
            else:
                return type(optimizer).__bases__[0]
        else:
            raise FedbiomedOptimizerError(f"Cannot find parent class of Optimizer {optimizer}")