# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""API and wrappers to interface framework-specific and generic optimizers."""

from abc import ABCMeta, abstractmethod
from types import TracebackType
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

import declearn
import declearn.model.torch
import numpy as np
import torch

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.logger import logger
from fedbiomed.common.models import Model, SkLearnModel, TorchModel
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer


OT = TypeVar("OT")  # generic type-annotation for wrapped optimizers
"""Generic TypeVar for framework-specific Optimizer types"""

class SklearnOptimizerProcessing:
    """Context manager used for scikit-learn model, that checks if model parameter(s) has(ve) been changed
    when disabling scikit-learn internal optimizer - ie when calling `disable_internal_optimizer` method

    """
    _model: SkLearnModel
    _disable_internal_optimizer: bool

    def __init__(self, model: SkLearnModel,
                 is_declearn_optimizer: bool):
        """Constructor of the object. Sets internal variables

        Args:
            model: a SkLearnModel that wraps a scikit-learn model
            is_declearn_optimizer: whether to disable scikit-learn model internal optimizer (True) in order
                to apply declearn one or to keep it (False)
        """
        self._model = model
        self._is_declearn_optimizer = is_declearn_optimizer

    def __enter__(self) -> 'SklearnOptimizerProcessing':
        """Called when entering context manager"""
        if self._is_declearn_optimizer:
            self._model.disable_internal_optimizer()

    def __exit__(
                self,
                type: Union[type[BaseException], None],
                value: Union[BaseException, None],
                traceback: Union[TracebackType, None]):
        """Called when leaving context manager.

        Args:
            type: default argument for `__exit__` method in context manager. Unused.
            value: default argument for `__exit__` method in context manager. Unused.
            traceback: default argument for `__exit__` method in context manager. Unused.
        """
        self._model.enable_internal_optimizer()
        # if self._disable_internal_optimizer:

        #     self._model.disable_internal_optimizer()
        #     # FIXME: this method `model_args` seems a bit ill to me
        #     is_param_changed, param_changed = self._model.check_changed_optimizer_params(self._model.model_args)
        #     if is_param_changed:
        #         msg = "The following parameter(s) has(ve) been detected in the model_args but will be disabled when using a declearn Optimizer: please specify those values in the training_args or in the init_optimizer method"
        #         msg += "\nParameters changed:\n"
        #         msg += param_changed
        #         logger.warning(msg)


class BaseOptimizer(Generic[OT], metaclass=ABCMeta):
    """Abstract base class for Optimizer and Model wrappers."""

    _model_cls: Union[Type[Model], Type[SkLearnModel]]

    def __init__(self, model: Model, optimizer: OT):
        """Constuctor of the optimizer wrapper that sets a reference to model and optimizer.

        Args:
            model: model to train, interfaced via a framework-specific Model.
            optimizer: optimizer that will be used for optimizing the model.

        Raises:
            FedbiomedOptimizerError:
                Raised if model is not an instance of `_model_cls` (which may
                be a subset of the generic Model type).
        """
        if not isinstance(model, self._model_cls):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621_b.value}, in `model` argument, expected an instance "
                f"of {self._model_cls} but got an object of type {type(model)}."
            )
        self._model: Model = model
        self.optimizer: OT = optimizer

    def init_training(self):
        """Sets up training and misceallenous parameters so the model is ready for training
        """
        self._model.init_training()

    def train_model(self,
                    inputs: Union[torch.Tensor, np.ndarray],
                    target: Union[torch.Tensor, np.ndarray],
                    **kwargs):
        """Performs a training of the model

        Args:
            inputs: inputs data
            target: targeted data
        """
        self._model.train(inputs, target, **kwargs)

    @abstractmethod
    def step(self):
        """Performs an optimisation step and updates model weights.
        """

    @abstractmethod
    def get_learning_rate(self)-> List[float]:
        """Returns learning rate(s) of the optimizer.

        !!! warning
            For pytorch and scikit-learn native optimizers, returned learning rate might be
            initial learning rate, and not the actual learning rate got over several model iteration.

        Returns:
            List[float]: learning-rate(s) contained in a list. Size of list
                depends of the optimizer specifed: if pytorch optimizer with several learning rates,
                size is of the number of layers contained in the model, size is of one otherwise.
        """


class BaseDeclearnOptimizer(BaseOptimizer, metaclass=ABCMeta):
    """Base Optimizer subclass to use a declearn-backed Optimizer."""

    def __init__(self, model: Model, optimizer: Union[FedOptimizer, declearn.optimizer.Optimizer]):
        """Constructor of Optimizer wrapper for declearn's optimizers

        Args:
            model: Model that wraps the actual model
            optimizer: declearn optimizer,
                or fedbiomed optimizer (that wraps declearn optimizer)
        """
        logger.debug("Using declearn optimizer")
        if isinstance(optimizer, declearn.optimizer.Optimizer):
            # convert declearn optimizer into a fedbiomed optimizer wrapper
            optimizer = FedOptimizer.from_declearn_optimizer(optimizer)
        elif not isinstance(optimizer, FedOptimizer):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621_b.value}: expected a declearn optimizer,"
                f" but got an object with type {type(optimizer)}."
            )
        super().__init__(model, optimizer)
        self.optimizer.init_round()

    def step(self):
        """Performs one optimization step"""
        # NOTA: for sklearn, gradients retrieved are unscaled because we are using learning rate equal to 1.
        # Therefore, it is necessary to disable the sklearn optimizer beforehand
        # otherwise, computation will be incorrect
        grad = declearn.model.api.Vector.build(self._model.get_gradients())
        weights = declearn.model.api.Vector.build(self._model.get_weights())
        updates = self.optimizer.step(grad, weights)
        self._model.apply_updates(updates.coefs)

    def get_learning_rate(self) -> List[float]:
        """Returns current learning rate of the optimizer

        Returns:
            a list containing the learning rate value (of size 1)
        """
        states = self.optimizer.get_state()['config']
        return [states['lrate']]

    def set_aux(self, aux: Dict[str, Any]):
        # attention: for imported tensors in PyTorch sent as auxiliary variables,
        # we should push it on the appropriate device (ie cpu/gpu)
        self.optimizer.set_aux(aux)

    def get_aux(self) -> Optional[Dict[str, Any]]:
        aux = self.optimizer.get_aux()
        return aux

    @classmethod
    def load_state(cls, model: Model, optim_state: Dict) -> 'BaseDeclearnOptimizer':
        # state: breakpoint content for optimizer
        relaoded_optim = FedOptimizer.load_state(optim_state)
        return cls(model, relaoded_optim)

    def save_state(self) -> Dict:
        optim_state = self.optimizer.get_state()
        return optim_state


class DeclearnTorchOptimizer(BaseDeclearnOptimizer):
    """Optimizer wrapper for declearn optimizers applied to pytorch models
    """
    _model_cls: Type[TorchModel] = TorchModel
    def zero_grad(self):
        """Zeroes gradients of the Pytorch model. Basically calls the `zero_grad`
        method of the model.

        Raises:
            FedbiomedOptimizerError: triggered if model has no method called `zero_grad`
        """
        # warning: specific for pytorch
        try:
            self._model.model.zero_grad()
        except AttributeError as err:
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB621_b.value} Model has no method named `zero_grad`: are you sure you are using a PyTorch TrainingPlan?."
                                          f"Details {repr(err)}") from err


class DeclearnSklearnOptimizer(BaseDeclearnOptimizer):
    """Optimizer wrapper for declearn optimizers applied to sklearn models
    """
    model_args: Dict[str, Any]
    _model_cls: Type[SkLearnModel] = SkLearnModel
    def __init__(self, model: SkLearnModel, optimizer: Union[FedOptimizer, declearn.optimizer.Optimizer, None]):
        super().__init__(model, optimizer)
        self.model_args = {}

    def optimizer_processing(self) -> SklearnOptimizerProcessing:
        """Provides a context manager able to do some actions before and after setting up an Optimizer, mainly disabling scikit-learn
        internal optimizer. Also, checks if `model_args` dictionary contains training parameters that
        won't be used or have any effect on the training, because of disabling the scikit-learn optimizer (
        such as initial learning rate, learnig rate scheduler, ...). If disabling the internal optimizer leads
        to such changes, displays a warning.

        Args:
            model_args: model_args sent by `Researcher` that instantiates a scikit-learn model.
                Should contain a mapping of scikit-learn parameter(s) and its(their) value(s).

        Returns:
            SklearnOptimizerProcessing: context manager providing extra logic

        Usage:
        ```python
            >>> dlo = DeclearnSklearnOptimizer(model, optimizer)
            >>> with dlo.optimizer_processing():
                    model.train(inputs,targets)
        ```
        """
        return SklearnOptimizerProcessing(self._model, is_declearn_optimizer=True)


class NativeTorchOptimizer(BaseOptimizer):
    """Optimizer wrapper for pytorch native optimizers and models.
    """
    _model_cls: Type[TorchModel] = TorchModel
    def __init__(self, model: TorchModel, optimizer: torch.optim.Optimizer):
        """Constructor of the optimizer wrapper

        Args:
            model: fedbiomed model wrapper that warps the pytorch model
            optimizer: pytorch native optimizers (inhereting from `torch.optim.Optimizer`)

        Raises:
            FedbiomedOptimizerError: raised if optimizer is not a pytorch native optimizer ie a `torch.optim.Optimizer` object.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB621_b.value} Expected a native pytorch `torch.optim` optimizer, but got {type(optimizer)}")
        super().__init__(model, optimizer)
        logger.debug("using native torch optimizer")

    def step(self):
        """Performs an optimization step and updates model weights
        """
        self.optimizer.step()

    def zero_grad(self):
        """Zeroes gradients of the Pytorch model. Basically calls the `zero_grad`
        method of the optimizer.
        """
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

    def fed_prox(self, loss: torch.float, mu: Union[float, 'torch.float']) -> 'torch.float':
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

        for current_model, init_model in zip(self._model.model().parameters(), self._model.init_params):
            norm += ((current_model - init_model) ** 2).sum()
        return norm


class NativeSkLearnOptimizer(BaseOptimizer):
    """Optimizer wrapper for scikit-learn native models.
    """
    model_args: Dict[str, Any]
    _model_cls: Type[SkLearnModel] = SkLearnModel
    def __init__(self, model: SkLearnModel, optimizer: Optional[None] = None):
        """Constructor of the Optimizer wrapper for scikit-learn native models.

        Args:
            model: SkLearnModel model that builds a scikit-learn model.
            optimizer: unused. Defaults to None.
        """
        # if not isinstance(model, SkLearnModel):
        #     raise FedbiomedOptimizerError(f"{ErrorNumbers.FB621_b.value} In model argument: expected a `SkLearnModel` object, but got {type(model)}")

        if optimizer is  not None:
            logger.info(f"Passed Optimizer {optimizer} won't be used (using only native scikit learn optimization)")
        super().__init__(model, None)
        self.model_args = {}
        logger.debug("Using native Sklearn Optimizer")

    def step(self):
        """Performs an optimization step and updates model weights
        """
        gradients = self._model.get_gradients()
        #lrate = self.model.get_learning_rate()[0]
        # revert back gradients to the batch averaged gradients
        #gradients: Dict[str, np.ndarray] = {layer: val * lrate for layer, val in gradients.items()}

        self._model.apply_updates(gradients)

    def optimizer_processing(self):
        return SklearnOptimizerProcessing(self._model, is_declearn_optimizer=False)

    def get_learning_rate(self) -> List[float]:
        """Returns scikit-learn model initial learning rate.

        !!! warning
            This method does not return the current learning rate, but the initial learning rate

        Returns:
            List[float]: initial learning rate of the model
        """
        return self._model.get_learning_rate()

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
    """Optimizer wrapper builder that creates the appropriate Optimizer given the nature of
    the optimizer.

    Usage:
    Example for plain Pytorch model
    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> opt_builder = OptimizerBuilder()
    >>> model = nn.Linear(4,2)
    >>> optimizer = torch.optim.SGD(.1, model.paramaters())
    >>> optim_wrapper = opt_builder.build(TrainingPlans.TorchTrainingPlan,
                                          model, optimizer)
    >>> optim_wrapper
        NativeTorchOptimizer
    ```
    """
    TORCH_OPTIMIZERS = {
    FedOptimizer: DeclearnTorchOptimizer,
    declearn.optimizer.Optimizer: DeclearnTorchOptimizer,
    torch.optim.Optimizer: NativeTorchOptimizer
}

    SKLEARN_OPTIMIZERS = {
        FedOptimizer: DeclearnSklearnOptimizer,
        declearn.optimizer.Optimizer: DeclearnSklearnOptimizer,
        None: NativeSkLearnOptimizer
    }

    def __init__(self) -> None:
        """Constructor of the builder: sets some internal variables
        """
        self.builder = {
            TrainingPlans.TorchTrainingPlan: self.build_torch,
            TrainingPlans.SkLearnTrainingPlan: self.build_sklearn
            }

    @staticmethod
    def build_torch(model: TorchModel,
                    optimizer: Union[FedOptimizer,
                                     torch.optim.Optimizer,
                                     declearn.optimizer.Optimizer]) -> Union[NativeTorchOptimizer,
                                                                             DeclearnTorchOptimizer]:
        """Builds Pytorch optimizer wrapper.

        Args:
            model: model wrapper that contains Pytorch model
            optimizer: either Fed-BioMed Optimizer wrapper (wrapping declearn's optimizer),
                a plain declearn optimizer or plain pytorch optimizer.

        Raises:
            FedbiomedOptimizerError: raised if Optimizer is not handled by the builder.

        Returns:
            Union[NativeTorchOptimizer, DeclearnTorchOptimizer]: Built Generic Optimizer,
                that contains a TorchModel and a Optimizer.
        """
        try:
            optimizer_wrapper: BaseOptimizer = OptimizerBuilder.TORCH_OPTIMIZERS[OptimizerBuilder.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"{ErrorNumbers.FB621_b.value} Optimizer {optimizer} is not compatible with training plan {TrainingPlans.TorchTrainingPlan.value}"

            raise FedbiomedOptimizerError(err_msg)
        return optimizer_wrapper(model, optimizer)

    @staticmethod
    def build_sklearn(model: SkLearnModel,
                      optimizer: Union[FedOptimizer,
                                       declearn.optimizer.Optimizer,
                                       None]) -> Union[NativeTorchOptimizer,
                                                       DeclearnTorchOptimizer]:
        """Builds scikit-learn optimizer wrapper.

        Args:
            model: Scikit-learn wrapper model that contains scikit-learn model.
            optimizer: either a declearn optimizer, a Fed-BioMed Optimizer wrapper that
                wraps a declearn Optimizer, or None object (meaning only scikit-learn internal
                optimizer will be used for updating the model).

        Raises:
            FedbiomedOptimizerError: raised if Optimizer is not handled by the builder.

        Returns:
            Union[NativeTorchOptimizer, DeclearnTorchOptimizer]: Built Generic Optimizer,
                that contains a SkLearnModel and a Optimizer (or None).
        """
        if not isinstance(model, SkLearnModel):
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB621_b.value} in `model` argument. Expected a SkLearnModel object but got {model}")
        try:
            optimizer_wrapper: BaseOptimizer = OptimizerBuilder.SKLEARN_OPTIMIZERS[OptimizerBuilder.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"{ErrorNumbers.FB621_b.value} Optimizer {optimizer} is not compatible with training plan {TrainingPlans.SkLearnTrainingPlan}" + \
            "\nHint: If If you want to use only native scikit learn optimizer, please do not define a `init_optimizer` method in the TrainingPlan"

            raise FedbiomedOptimizerError(err_msg)
        return optimizer_wrapper(model, optimizer)

    def build(self, tp_type: TrainingPlans, model: Model, optimizer: Optional[Union[torch.optim.Optimizer, FedOptimizer]]=None) -> 'BaseOptimizer':
        """Builds a Optimizer wrapper based on TrainingPlans and optimizer type

        Args:
            tp_type: training plan type
            model: model wrapper that contains a model. Must be compatible with the training plan type.
            optimizer: optimizer used to optimize model. For using plain Scikit-Learn model, should be set to None (meaning only model optimization will be used). Defaults to None.

        Raises:
            FedbiomedOptimizerError: raised if training plan type `tp_type` and `optimizer`
                are not compatible (eg using torch optimizer with sklearn models)

        Returns:
            BaseOptimizer: an instance of a child of BaseOptimizer object
        """
        # if self._optimizers_available is None:
        #     raise FedbiomedOptimizerError("error, no training_plan set, please run `set_optimizers_for_training_plan` beforehand")

        try:
            return self.builder[tp_type](model, optimizer)
            #optimizer_wrapper: BaseOptimizer = self._optimizers_available[self.get_parent_class(optimizer)]
        except KeyError:
            err_msg = f"{ErrorNumbers.FB621_b.value} Unknown Training Plan type {tp_type} "

            raise FedbiomedOptimizerError(err_msg)

    @staticmethod
    def get_parent_class(optimizer: Union[None, Any]) -> Union[None, Type]:
        """Gets parent class type of the instance of the class passed (class just after `object` class)
        If class is already at top level, returns the class itself.

        Args:
            optimizer: child class from which we want to extract the parent class.
                If None is passed, returns None.

        Raises:
            FedbiomedOptimizerError: raised when failing to retrieve parent class. This could
                happen if some objects have some of their magic methods modified, such as `__bases__`
                magic method.

        Returns:
            Union[None, Type]: the parent class type, or the class itself, if class is just under `object`
        """
        if optimizer is None:
            return None
        if hasattr(type(optimizer), '__bases__'):
            if type(optimizer).__bases__[0]  is object:
                # in this case, `optimizer` is already the parent class (it only has `object`as parent class)
                return type(optimizer)
            else:
                return type(optimizer).__bases__[0]
        else:
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB621_b.value} Cannot find parent class of Optimizer {optimizer}")
