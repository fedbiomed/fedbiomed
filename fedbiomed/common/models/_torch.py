# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Tuple, Union


from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.models import Model

import numpy as np

from declearn.model.torch import TorchVector

import torch



class TorchModel(Model):
    """PyTorch model wrapper that ease the handling of a pytorch model
    
    Attributes:
        model: torch.nn.Module. Pytorch model wrapped.
        init_params: OrderedDict. Model initial parameters. Set when calling `init_training`
    """

    model: torch.nn.Module
    init_params: OrderedDict

    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiates the wrapper over a torch Module instance."""
        # if not isinstance(model, torch.nn.Module):
        #     raise FedbiomedModelError(f"invalid argument for `model`: expecting a torch.nn.Module, but got {type(model)}")
        super().__init__(model)
        self.init_params = None

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
                non-frozen layers) or all layers (trainable and frozen). Defaults to False, (trainable and
                frozen ones) 
            return_type (Callable, optional): callable that loads weights into a 
                data structure and outputs weights in this data structure. If not provided,
                returns weights under a dictionary mapping model's layer names to theirs tensors. 
                Defaults to None. 
        
        Returns:
            Model's weights in a dictionary mapping model's layer names to theirs tensors
                (I am going to change that if `return_type` argument is not provided) or in
                a data structure returned by `return_type` Callable.
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
            updates: model updates to be added to the model.
        """

        iterator = self._get_iterator_model_params(updates)
        with torch.no_grad():
            for name, update in iterator:
                param = self.model.get_parameter(name)
                param.add_(update)
    
    def add_corrections_to_gradients(self, corrections: Union[TorchVector, Dict[str, torch.Tensor]]):
        """Adds values to attached gradients in the model
        
        Args:
            corrections: corrections to be added to model's gradients

        """
        iterator = self._get_iterator_model_params(corrections)

        for name, update in iterator:
            param = self.model.get_parameter(name)
            if param.grad is not None:
                param.grad.add_(update.to(param.grad.device))

    @staticmethod
    def _get_iterator_model_params(model_params: Union[Dict[str, torch.Tensor], TorchVector]) -> Iterable[Tuple[str, torch.Tensor]]:
        """Returns an iterable from model_params, whether it is a 
        dictionary or a declearn's TorchVector

        Args:
            model_params: model parameters

        Raises:
            FedbiomedModelError: raised if argument `model_params` type is neither
            a TorchVector nor a dictionary

        Returns:
            Iterable[Tuple]: iterable containing model parameters, that returns layer name and its value
        """
        if isinstance(model_params, TorchVector):
            
            iterator = model_params.coefs.items()
        elif isinstance(model_params, dict):
            iterator = model_params.items()
        else:
            raise FedbiomedModelError(f"{ErrorNumbers.FB622.value}. Got a {type(model_params)} while expecting TorchVector or OrderedDict/Dict")
        return iterator

    def predict(self, inputs: torch.Tensor)-> np.ndarray:
        """Computes prediction given input data.

        Args:
            inputs: input data

        Returns:
            Model predictions returned as a numpy array
        """
        self.model.eval()  # pytorch switch for model inference-mode
        with torch.no_grad():
            pred = self.model(inputs) 
        return pred.cpu().numpy()
    
    def send_to_device(self, device: torch.device):
        """Sends model to device
        
        Args:
            device: device set for using GPU or CPU.
        """
        return self.model.to(device)

    def init_training(self):
        """Initializes and sets attributes before the training.

        Initializes `init_params` as a copy of the initial parameters of the model
        """
        # initial aggregated model parameters
        self.init_params = deepcopy(list(self.model.parameters()))
        self.model.train()  # pytorch switch for training
        self.model.zero_grad()
        
    def train(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs):
        # TODO: should we pass loss function here? and do the backward prop?
        if self.init_params is None:
            raise FedbiomedModelError(f"{ErrorNumbers.FB622.value}. Training has not been initialized, please initalize it beforehand")
        pass

    def load(self, filename: str) -> OrderedDict:
        # loads model from a file
        params = torch.load(filename)
        self.model.load_state_dict(params)
        return params
        
    def save(self, filename: str):
        torch.save(self.model.state_dict(), filename)