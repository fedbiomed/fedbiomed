# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.models import SkLearnModel
from fedbiomed.common.optimizers.generic_optimizers import BaseOptimizer, OptimizerBuilder
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common import utils

from ._base_training_plan import BaseTrainingPlan


class SKLearnTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this abstract class must:
    - Specify a `_model_cls` class attribute that defines the type
      of scikit-learn model being wrapped for training.
    - Implement a `set_init_params` method that:
      - sets and assigns the model's initial trainable weights attributes.
      - populates the `_param_list` attribute with names of these attributes.
    - Implement a `_training_routine` method that performs a training round
      based on `self.train_data_loader` (which is a `NPDataLoader`).

    Attributes:
        dataset_path: The path that indicates where dataset has been stored
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.

    !!! info "Notes"
        The trained model may be exported via the `export_model` method,
        resulting in a dump file that may be reloded using `joblib.load`
        outside of Fed-BioMed.
    """

    _model_cls: Type[BaseEstimator]  # wrapped model class
    _model_dep: Tuple[str, ...] = tuple()  # model-specific dependencies

    def __init__(self) -> None:
        """Initialize the SKLearnTrainingPlan."""
        super().__init__()
        self._model: Union[SkLearnModel, None] = None
        self._training_args = {}  # type: Dict[str, Any]
        self.__type = TrainingPlans.SkLearnTrainingPlan
        self._batch_maxnum = 0
        self.dataset_path: Optional[str] = None
        self._optimizer: Optional[BaseOptimizer] = None
        self._add_dependency([
            "import inspect",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
            "from fedbiomed.common.data import DataManager",
        ])
        self._add_dependency(list(self._model_dep))

        # Add dependencies
        self._configure_dependencies()

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
            initialize_optimizer: bool = True
    ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
            initialize_optimizer: Unused.
        """
        model_args.setdefault("verbose", 1)
        super().post_init(model_args, training_args, aggregator_args)
        self._model = SkLearnModel(self._model_cls)
        self._batch_maxnum = self._training_args.get('batch_maxnum', self._batch_maxnum)
        self._warn_about_training_args()

        # configure optimizer (if provided in the TrainingPlan)
        self._configure_optimizer()

        # FIXME: should we do that in `_configure_optimizer`
        # from now on, `self._optimizer`` is not None
        # Override default model parameters based on `self._model_args`.
        params = {
            key: model_args.get(key, val)
            for key, val in self._model.get_params().items()
        }
        self._model.set_params(**params)
        # Set up additional parameters (normally created by `self._model.fit`).
        self._model.set_init_params(model_args)

    def set_data_loaders(
            self,
            train_data_loader: Union[DataLoader, NPDataLoader, None],
            test_data_loader: Union[DataLoader, NPDataLoader, None]
    ) -> None:
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for validation routine
        """
        args = (train_data_loader, test_data_loader)
        if not all(isinstance(data, NPDataLoader) for data in args):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan expects "
                "NPDataLoader instances as training and testing data "
                f"loaders, but received {type(train_data_loader)} "
                f"and {type(test_data_loader)} respectively."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def model(self) -> Optional[BaseEstimator]:
        """Retrieve the wrapped scikit-learn model instance.

        Returns:
            Scikit-learn model instance
        """
        if self._model is not None:
            return self._model.model
        else:
            return self._model

    def _configure_optimizer(self):
        """Configures declearn Optimizer for scikit-learn if method
        `init_optimizer` is provided in the TrainingPlan, otherwise considers
        only scikit-learn internal optimization.

        Raises:
            FedbiomedTrainingPlanError: raised if no model has been found
            FedbiomedTrainingPlanError: raised if more than one argument has been provided
                to the `init_optim` method.
        """
        # Message to format for unexpected argument definitions in special methods
        method_error = \
            ErrorNumbers.FB605.value + ": Special method `{method}` has more than one argument: {keys}. This method " \
                                       "can not have more than one argument/parameter (for {prefix} arguments) or " \
                                       "method can be defined without argument and `{alternative}` can be used for " \
                                       "accessing {prefix} arguments defined in the experiment."

        if self._model is None:
            raise FedbiomedTrainingPlanError("can not configure optimizer, Model is None")
        # Get optimizer defined by researcher ---------------------------------------------------------------------
        init_optim_spec = utils.get_method_spec(self.init_optimizer)
        if not init_optim_spec:
            optimizer = self.init_optimizer()
        elif len(init_optim_spec.keys()) == 1:
            optimizer = self.init_optimizer(self.optimizer_args())
        else:
            raise FedbiomedTrainingPlanError(method_error.format(prefix="optimizer",
                                                                 method="init_optimizer",
                                                                 keys=list(init_optim_spec.keys()),
                                                                 alternative="self.optimizer_args()"))
        # create optimizer builder
        optim_builder = OptimizerBuilder()

        # then build optimizer wrapper given model and optimizer
        self._optimizer = optim_builder.build(self.__type, self._model, optimizer)

    def init_optimizer(self) -> Optional[FedOptimizer]:
        """Creates and configures optimizer. By default, returns None (meaning native inner scikit
        learn optimization SGD based will be used).

        In the case a Declearn Optimizer is used, this method should be overridden in the Training Plan and return
        a Fedbiomed [`Optimizer`][fedbiomed.common.optimizers.optimizer.Optimizer]"""
        pass

    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Training routine, to be called once per round.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording training metadata. Defaults to None.
            node_args: command line arguments for node.
                These arguments can specify GPU use; however, this is not
                supported for scikit-learn models and thus will be ignored.
        """
        if self._optimizer is None:
            raise FedbiomedTrainingPlanError('Optimizer is None, please run `post_init` beforehand')

        # Run preprocesses
        self._preprocess()

        if not isinstance(self.training_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot "
                "be trained without a NPDataLoader as `training_data_loader`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):

            self._optimizer.send_to_device(False)  # disable GPU, avoid `declearn` triggering warning messages
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )
        # Run the model-specific training routine.
        try:
            return self._training_routine(history_monitor)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while fitting "
                f"the model: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    @abstractmethod
    def _training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None
    ) -> None:
        """Model-specific training routine backend.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording the loss value during training.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`
        (that notably enforces preprocessing and exception catching).
        """

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional['HistoryMonitor'],
            before_train: bool
    ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        # Check that the testing data loader is of proper type.
        if not isinstance(self.testing_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot be "
                "evaluated without a NPDataLoader as `testing_data_loader`."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        # If required, make up for the lack of specifications regarding target
        # classification labels.
        if self._model.is_classification and not hasattr(self.model(), 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self.model(), 'classes_', classes)
        # If required, select the default metric (accuracy or mse).
        if metric is None:
            if self._model.is_classification:
                metric = MetricTypes.ACCURACY
            else:
                metric = MetricTypes.MEAN_SQUARE_ERROR
        # Delegate the actual evalation routine to the parent class.
        super().testing_routine(
            metric, metric_args, history_monitor, before_train
        )

    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """Return unique target labels from the training and testing datasets.

        Returns:
            Numpy array containing the unique values from the targets wrapped
            in the training and testing NPDataLoader instances.
        """
        return np.unique([t for loader in (self.training_data_loader, self.testing_data_loader) for d, t in loader])

    def type(self) -> TrainingPlans:
        """Getter for training plan type """
        return self.__type

    def _warn_about_training_args(self):
        if self._training_args['share_persistent_buffers']:
            logger.warning("Option share_persistent_buffers is not supported in SKLearnTrainingPlan, "
                           "it will be ignored.")
