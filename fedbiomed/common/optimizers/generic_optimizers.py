# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""API and wrappers to interface framework-specific and generic optimizers."""

from abc import ABCMeta, abstractmethod
import copy
from types import TracebackType
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import declearn
import declearn.model.torch
import torch

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.logger import logger
from fedbiomed.common.models import Model, SkLearnModel, TorchModel
from fedbiomed.common.optimizers.declearn import AuxVar
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
                f"{ErrorNumbers.FB626.value}, in `model` argument, expected an instance "
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

    def load_state(self, optim_state: Dict, load_from_state: bool = False) -> Union['BaseOptimizer', None]:
        """Reconfigures optimizer from a given state.

        This is the default method for optimizers that don't support state. Does nothing.

        Args:
            optim_state: not used
            load_from_state (optional): not used

        Returns:
            None
        """
        logger.warning("load_state method of optimizer not implemented, cannot load optimizer status")
        return None

    def save_state(self) -> Union[Dict, None]:
        """Gets optimizer state.

        This is the default method for optimizers that don't support state. Does nothing.

        Returns:
            None
        """
        logger.warning("save_state method of optimizer not implemented, cannot save optimizer status")
        return None

    def send_to_device(self, device: str, idx: Optional[int] = None):
        """GPU support"""

    def count_nb_auxvar(self) -> int:
        """Counts number of auxiliary variables needed for the given optimizer"""
        return 0


class DeclearnOptimizer(BaseOptimizer):
    """Base Optimizer subclass to use a declearn-backed Optimizer."""
    _model_cls: Tuple[Type] = (TorchModel, SkLearnModel)
    optimizer = None
    #model = None


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
                f"{ErrorNumbers.FB626.value}: expected a declearn optimizer,"
                f" but got an object with type {type(optimizer)}."
            )
        super().__init__(model, optimizer)
        #self.optimizer.init_round()

    def init_training(self):
        super().init_training()
        self.optimizer.init_round()

    def step(self):
        """Performs one optimization step"""
        # NOTA: for sklearn, gradients retrieved are unscaled because we are using learning rate equal to 1.
        # Therefore, it is necessary to disable the sklearn internal optimizer beforehand
        # otherwise, computation will be incorrect
        grad = declearn.model.api.Vector.build(self._model.get_gradients())
        weights = declearn.model.api.Vector.build(self._model.get_weights(
            only_trainable=False,
            exclude_buffers=True
        ))
        updates = self.optimizer.step(grad, weights)
        self._model.apply_updates(updates.coefs)

    def set_aux(self, aux: Dict[str, AuxVar]):
        # FIXME: for imported tensors in PyTorch sent as auxiliary variables,
        # we should push it on the appropriate device (ie cpu/gpu)
        # TODO-PAUL: call the proper declearn routines
        self.optimizer.set_aux(aux)

    def get_aux(self) -> Optional[Dict[str, AuxVar]]:
        aux = self.optimizer.get_aux()
        return aux

    def count_nb_auxvar(self) -> int:
        return len(self.optimizer.get_aux_names())

    def load_state(self, optim_state: Dict[str, Any], load_from_state: bool = False) -> 'DeclearnOptimizer':
        """Reconfigures optimizer from a given state (contained in `optim_state` argument).
        Usage:
        ```python
        >>> import torch.nn as nn
        >>> from fedbiomed.common.optimizers import Optimizer
        >>> from fedbiomed.common.models import TorchModel
        >>> model = TorchModel(nn.Linear(4, 2))
        >>> optimizer = Optimizer(lr=.1)
        >>> optim = DeclearnOptimizer(model, optimizer)

        >>> optim.load_state(state)  # provided state contains the state one wants to load the optimizer with
        ```
        If `load_from_state` argument is True, it completes the current optimizer state with `optim_state` argument

        ```python
        >>> import torch.nn as nn
        >>> from fedbiomed.common.optimizers import Optimizer
        >>> from fedbiomed.common.optimizers.declearn import MomentumModule, AdamModule
        >>> from fedbiomed.common.models import TorchModel
        >>> model = TorchModel(nn.Linear(4, 2))
        >>> optimizer = Optimizer(lr=.1, modules=[MomentumModule(), AdamModule()])
        >>> optim_1 = DeclearnOptimizer(model, optimizer)

        >>> optimizer = Optimizer(lr=.1, modules=[AdamModule(), MomentumModule()])
        >>> optim_2 = DeclearnOptimizer(model, optimizer)
        >>> optim_2.load_state(optim_1.save_state())
        >>> optim_2.save_state()['states']
        {'modules': [('momentum', {'velocity': 0.0}),
                ('adam',
                {'steps': 0,
                    'vmax': None,
                    'momentum': {'state': 0.0},
                    'velocity': {'state': 0.0}})]}
        ```
        Modules of DeclearnOptimizer will be reloaded provided that Module is the same and occupying the same index.
        Eg if the state contains following modules:
        ```modules=[AdamModule(), AdagradModule(), MomemtumModule()]```
         And the Optimizer contained in the TrainingPlan has the following modules:
        ```modules=[AdamModule(), MomemtumModule()]```
        Then only `AdamModule` module will be reloaded, `MomentumModule` will be set with default argument (they don't
        share the same index in the modules list).

        Args:
            optim_state: state of the Optimizer to be loaded. It will change the current state of the optimizer
                with the one loaded
            load_from_state (optional): strategy for loading states: whether to load from saved states (True) or
                from breakpoint (False).
                If set to True, loading is done partially in the sense that if some of the OptimModules is different in
                the optim_state and the original state of the optimizer, it loads only the OptiModule(s) from the
                latest state that both state has in common. Defaults to False.

        Raises:
            FedbiomedOptimizerError: raised if state is not of dict type.

        Returns:
            Optimizer wrapper reloaded from `optim_state` argument.
        """
        # state: breakpoint content for optimizer
        if not isinstance(optim_state, Dict):
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB626.value}, incorrect type of argument `optim_state`: "
                                          f"expecting a dict, but got {type(optim_state)}")

        if load_from_state:
            # first get Optimizer detailed in the TrainingPlan.

            init_optim_state = self.optimizer.get_state()  # we have to get states since it is the only way we can
            # gather modules (other methods of `Optimizer are private`)

            optim_state_copy = copy.deepcopy(optim_state)
            optim_state.update(init_optim_state)  # optim_state will be updated with current optimizer state
            # check if opimizer state has changed from last optimizer to the current one
            # if it has changed, find common modules and update common states
            for component in ( 'modules', 'regularizers',):
                components_to_keep: List[Tuple[str, int]] = []  # we store here common Module between current Optimizer
                # and the ones in the `optim_state` tuple (common Module name, index in List)

                if not init_optim_state['states'].get(component) or not optim_state_copy['states'].get(component):
                    continue
                self._collect_common_optimodules(
                    init_optim_state,
                    optim_state_copy,
                    component,
                    components_to_keep
                )

                for mod in components_to_keep:
                    for mod_state in optim_state_copy['states'][component]:
                        if mod[0] == mod_state[0]:
                            # if we do find same module in the current optimizer than the previous one,
                            # we load the previous optimizer module state into the current one
                            optim_state['states'][component][mod[1]] = mod_state

            logger.info("Loading optimizer state from saved state")

        reloaded_optim = FedOptimizer.load_state(optim_state)
        self.optimizer = reloaded_optim

        return self

    def _collect_common_optimodules(self,
                                    init_state: Dict,
                                    optim_state: Dict,
                                    component_name: str,
                                    components_to_keep: List[Tuple[str, int]]):
        """Methods that checks which modules and regularizers are common from `init_state`, the current state Optimizer
        state, and `optim_state`, the previous optimizer state. Populates in that regard the `components_to_keep` list,
        a list containing common Modules and Regularizers, and that can be access through its reference.

        Args:
            init_state: current state Optimizer state
            optim_state: previous optimizer state
            component_name: component string names that access the list of OptiModule or Regularizers
                (usually either 'modules' or 'regularizers')
            components_to_keep: list containing tuples of common modules between `init_state`
                and `optim_state`. Each list item has the following entry: (module_name, index of the list).
        """
        idx: int = 0  # list index

        for init_module, new_module in zip(init_state['states'][component_name],
                                           optim_state['states'][component_name]):
            if init_module[0] == new_module[0]:
                # if we have the same modules from last to current round, update module wrt last saved state
                components_to_keep.append((new_module[0], idx))
            idx += 1

    def save_state(self) -> Dict:
        """Gets optimizer state.

        Returns:
            optimizer state
        """
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
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB626.value}. This method can only be used for TorchModel, "
                                          f"but got {self._model}")
        self._model.model.zero_grad()

    def optimizer_processing(self) -> SklearnOptimizerProcessing:
        """Provides a context manager able to do some actions before and after setting up an Optimizer, mainly
        disabling scikit-learn internal optimizer.

        Also, checks if `model_args` dictionary contains training parameters that
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
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB626.value}: Method optimizer_processing should be used "
                                          f"only with SkLearnModel, but model is {self._model}")

    def send_to_device(self, device: str, idx: int | None = None):
        self.optimizer.send_to_device(device, idx)


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
            FedbiomedOptimizerError: raised if optimizer is not a pytorch native optimizer ie a `torch.optim.Optimizer`
                object.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise FedbiomedOptimizerError(f"{ErrorNumbers.FB626.value} Expected a native pytorch `torch.optim` "
                                          f"optimizer, but got {type(optimizer)}")
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

    def get_learning_rate(self) -> Dict[str, float]:
        """Gets learning rates from param groups in Pytorch optimizer.

        For each optimizer param group, it iterates over all parameters in that parameter group and searches for the "
        corresponding parameter of the model by iterating over all model parameters. If it finds a correspondence,
        it saves the learning rate value. This function assumes that the parameters in the optimizer and the model
        have the same reference.


        !!! warning
            This function gathers the base learning rate applied to the model weights,
            including alterations due to any LR scheduler. However, it does not catch
            any adaptive component, e.g. due to RMSProp, Adam or such.

        Returns:
            List[float]: list of single learning rate or multiple learning rates
                (as many as the number of the layers contained in the model)
        """
        logger.warning(
            "`get_learning_rate` is deprecated and will be removed in future Fed-BioMed releases",
            broadcast=True)

        mapping_lr_layer_name: Dict[str, float] = {}

        for param_group in self.optimizer.param_groups:
            for layer_params in param_group['params']:
                for layer_name, tensor in self._model.model.named_parameters():
                    if layer_params is tensor:
                        mapping_lr_layer_name[layer_name] = param_group['lr']
        return mapping_lr_layer_name


class NativeSkLearnOptimizer(BaseOptimizer):
    """Optimizer wrapper for scikit-learn native models."""

    _model_cls: Type[SkLearnModel] = SkLearnModel

    def __init__(self, model: SkLearnModel, optimizer: Optional[None] = None):
        """Constructor of the Optimizer wrapper for scikit-learn native models.

        Args:
            model: SkLearnModel model that builds a scikit-learn model.
            optimizer: unused. Defaults to None.
        """

        if optimizer is not None:
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
    >>> from fedbiomed.common.models import TorchModel
    >>> opt_builder = OptimizerBuilder()
    >>> model = TorchModel(nn.Linear(4,2))
    >>> optimizer = torch.optim.SGD(model.paramaters(), .1)
    >>> optim_wrapper = opt_builder.build(TrainingPlans.TorchTrainingPlan, model, optimizer)
    >>> optim_wrapper()
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
        optimizer: Optional[Union[torch.optim.Optimizer, FedOptimizer]] = None,
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
            err_msg = f"{ErrorNumbers.FB626.value} Unknown Training Plan type {tp_type} "
            raise FedbiomedOptimizerError(err_msg) from exc
        try:
            return selected_optim_wrapper[optim_cls](model, optimizer)

        except KeyError as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB626.value} Training Plan type {tp_type} not compatible with optimizer {optimizer}"
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
                f"{ErrorNumbers.FB626.value} Cannot find parent class of Optimizer {optimizer}"
            )
