# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""API and wrappers to interface framework-specific and generic optimizers."""

from abc import ABCMeta, abstractmethod
from types import TracebackType
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import declearn
import declearn.model.torch
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

    def __init__(
        self,
        model: SkLearnModel,
        disable_internal_optimizer: bool
    ) -> None:
        """Constructor of the object. Sets internal variables

        Args:
            model: a SkLearnModel that wraps a scikit-learn model
            disable_internal_optimizer: whether to disable scikit-learn model internal optimizer (True) in order
                to apply declearn one or to keep it (False)
        """
        self._model = model
        self._disable_internal_optimizer = disable_internal_optimizer

    def __enter__(self) -> None:
        """Called when entering context manager"""
        if self._disable_internal_optimizer:
            self._model.disable_internal_optimizer()

    def __exit__(
            self,
            type: Union[type[BaseException], None],
            value: Union[BaseException, None],
            traceback: Union[TracebackType, None],
        ) -> None:
        """Called when leaving context manager.

        Args:
            type: default argument for `__exit__` method in context manager. Unused.
            value: default argument for `__exit__` method in context manager. Unused.
            traceback: default argument for `__exit__` method in context manager. Unused.
        """
        self._model.enable_internal_optimizer()


class BaseOptimizer(Generic[OT], metaclass=ABCMeta):
    """Abstract base class for Optimizer and Model wrappers."""

    _model_cls: Union[Type[Model], Type[SkLearnModel], Tuple[Type]]

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
                f"{ErrorNumbers.FB625.value}, in `model` argument, expected an instance "
                f"of {self._model_cls} but got an object of type {type(model)}."
            )
        self._model: Model = model
        self.optimizer: OT = optimizer

    def init_training(self):
        """Sets up training and misceallenous parameters so the model is ready for training
        """
        self._model.init_training()

    @abstractmethod
    def step(self):
        """Performs an optimisation step and updates model weights.
        """


class DeclearnOptimizer(BaseOptimizer):
    """Base Optimizer subclass to use a declearn-backed Optimizer."""
    _model_cls: Tuple[Type] = (TorchModel, SkLearnModel)

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
                f"{ErrorNumbers.FB625.value}: expected a declearn optimizer,"
                f" but got an object with type {type(optimizer)}."
            )
        super().__init__(model, optimizer)
        self.optimizer.init_round()

    def step(self):
        """Performs one optimization step"""
        # NOTA: for sklearn, gradients retrieved are unscaled because we are using learning rate equal to 1.
        # Therefore, it is necessary to disable the sklearn internal optimizer beforehand
        # otherwise, computation will be incorrect
        grad = declearn.model.api.Vector.build(self._model.get_gradients())
        weights = declearn.model.api.Vector.build(self._model.get_weights())
        updates = self.optimizer.step(grad, weights)
        self._model.apply_updates(updates.coefs)

    def set_aux(self, aux: Dict[str, Any]):
        # FIXME: for imported tensors in PyTorch sent as auxiliary variables,
        # we should push it on the appropriate device (ie cpu/gpu)
        self.optimizer.set_aux(aux)

    def get_aux(self) -> Optional[Dict[str, Any]]:
        aux = self.optimizer.get_aux()
        return aux

    @classmethod
    def load_state(cls, model: Model, optim_state: Dict) -> 'DeclearnOptimizer':
        # state: breakpoint content for optimizer
        relaoded_optim = FedOptimizer.load_state(optim_state)
        return cls(model, relaoded_optim)

    def save_state(self) -> Dict:
        optim_state = self.optimizer.get_state()
        return optim_state

    # TorchModel specific methods

    def zero_grad(self):
        """Zeroes gradients of the Pytorch model. Basically calls the `zero_grad`
        method of the model.

        Raises:
            FedbiomedOptimizerError: triggered if model has no method called `zero_grad`
        """
        # warning: specific for pytorch
        if not isinstance(self._model, TorchModel):
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB625.value}. This method can only be used for TorchModel, but got {self._model}")
        self._model.model.zero_grad()

    def optimizer_processing(self) -> SklearnOptimizerProcessing:
        """Provides a context manager able to do some actions before and after setting up an Optimizer, mainly disabling scikit-learn
        internal optimizer. Also, checks if `model_args` dictionary contains training parameters that
        won't be used or have any effect on the training, because of disabling the scikit-learn optimizer (
        such as initial learning rate, learnig rate scheduler, ...). If disabling the internal optimizer leads
        to such changes, displays a warning.

        Returns:
            SklearnOptimizerProcessing: context manager providing extra logic

        Usage:
        ```python
            >>> dlo = DeclearnSklearnOptimizer(model, optimizer)
            >>> with dlo.optimizer_processing():
                    model.train(inputs,targets)
        ```
        """
        if isinstance(self._model, SkLearnModel):
            return SklearnOptimizerProcessing(self._model, disable_internal_optimizer=True)
        else:
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB625.value}: Method optimizer_processing should be used only with SkLearnModel, but model is {self._model}")


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
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB625.value} Expected a native pytorch `torch.optim` optimizer, but got {type(optimizer)}")
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
        logger.warning("`get_learning_rate` is deprecated and will be removed in future Fed-BioMed releases")
        learning_rates = []
        params = self.optimizer.param_groups
        for param in params:
            learning_rates.append(param['lr'])
        return learning_rates


class NativeSkLearnOptimizer(BaseOptimizer):
    """Optimizer wrapper for scikit-learn native models."""

    _model_cls: Type[SkLearnModel] = SkLearnModel

    def __init__(self, model: SkLearnModel, optimizer: Optional[None] = None):
        """Constructor of the Optimizer wrapper for scikit-learn native models.

        Args:
            model: SkLearnModel model that builds a scikit-learn model.
            optimizer: unused. Defaults to None.
        """

        if optimizer is  not None:
            logger.info(f"Passed Optimizer {optimizer} won't be used (using only native scikit learn optimization)")
        super().__init__(model, None)
        logger.debug("Using native Sklearn Optimizer")

    def step(self):
        """Performs an optimization step and updates model weights."""
        gradients = self._model.get_gradients()
        updates = {k: -v for k, v in gradients.items()}
        self._model.apply_updates(updates)

    def optimizer_processing(self) -> SklearnOptimizerProcessing:
        return SklearnOptimizerProcessing(self._model, disable_internal_optimizer=False)


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
        FedOptimizer: DeclearnOptimizer,
        declearn.optimizer.Optimizer: DeclearnOptimizer,
        torch.optim.Optimizer: NativeTorchOptimizer
    }

    SKLEARN_OPTIMIZERS = {
        FedOptimizer: DeclearnOptimizer,
        declearn.optimizer.Optimizer: DeclearnOptimizer,
        None: NativeSkLearnOptimizer
    }

    BUILDER = {
        TrainingPlans.TorchTrainingPlan: TORCH_OPTIMIZERS,
        TrainingPlans.SkLearnTrainingPlan: SKLEARN_OPTIMIZERS
    }

    def build(
        self,
        tp_type: TrainingPlans,
        model: Model,
        optimizer: Optional[Union[torch.optim.Optimizer, FedOptimizer]]=None,
    ) -> 'BaseOptimizer':
        """Builds a Optimizer wrapper based on TrainingPlans and optimizer type

        Args:
            tp_type: training plan type
            model: model wrapper that contains a model. Must be compatible with the training plan type.
            optimizer: optimizer used to optimize model. For using plain Scikit-Learn model, should be
                set to None (meaning only model optimization will be used). Defaults to None.

        Raises:
            FedbiomedOptimizerError: raised if training plan type `tp_type` and `optimizer`
                are not compatible (eg using torch optimizer with sklearn models)

        Returns:
            BaseOptimizer: an instance of a child of BaseOptimizer object
        """
        try:
            optim_cls = OptimizerBuilder.get_parent_class(optimizer)
            selected_optim_wrapper = OptimizerBuilder.BUILDER[tp_type]
        except KeyError as exc:
            err_msg = f"{ErrorNumbers.FB625.value} Unknown Training Plan type {tp_type} "
            raise FedbiomedOptimizerError(err_msg) from exc
        try:
            return selected_optim_wrapper[optim_cls](model, optimizer)
        except KeyError as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB625.value} Training Plan type {tp_type} not compatible with optimizer {optimizer}"
            ) from exc

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
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB625.value} Cannot find parent class of Optimizer {optimizer}"
            )
