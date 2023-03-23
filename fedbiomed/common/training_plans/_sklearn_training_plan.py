# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer
from fedbiomed.common.training_args import TrainingArgs
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from declearn.optimizer import Optimizer as DeclearnOptimizer

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers.generic_optimizers import BaseOptimizer, OptimizerBuilder, NativeSkLearnOptimizer
from fedbiomed.common.utils import get_method_spec
from fedbiomed.common.models import SkLearnModel

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
    """

    _model_cls: Type[BaseEstimator]        # wrapped model class
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
        self.add_dependency([
            "import inspect",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
            "from fedbiomed.common.data import DataManager",
        ])
        self.add_dependency(list(self._model_dep))

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
        """
        self._model = SkLearnModel(self._model_cls)
        model_args.setdefault("verbose", 1)
        self._model.model_args = model_args
        self._aggregator_args = aggregator_args or {}
        
        self._optimizer_args = training_args.optimizer_arguments() or {}
        self._training_args = training_args.pure_training_arguments()
        self._batch_maxnum = self._training_args.get('batch_maxnum', self._batch_maxnum)

        # Add dependencies
        self._configure_dependencies()
        # Override default model parameters based on `self._model_args`.
        params = {
            key: model_args.get(key, val)
            for key, val in self._model.get_params().items()
        }

        # configure optimizer (if provided in the TrainingPlan)
        self._configure_optimizer()
        
        # FIXME: should we do that in `_configure_optimizer`
        # from now on, `self._optimizer`` is not None
        self._model.set_params(**params)
        # Set up additional parameters (normally created by `self._model.fit`).

        self._model.set_init_params(model_args)
        
        self._optimizer.optimizer_post_processing(model_args)
        
        # if isinstance(self._optimizer, NativeSkLearnOptimizer):
        #     # disable internal optimizer if optimizer is non native (ie declearn optimizer)
        #     self._optimizer.model.disable_internal_optimizer()
        #     is_param_changed, param_changed = self._optimizer.model.check_changed_optimizer_params(model_args)
        #     if is_param_changed:
        #         msg = "The following parameter(s) has(ve) been detected in the model_args but will be disabled when using a declearn Optimizer: please specify those values in the training_args or in the init_optimizer method"
        #         msg += "\nParameters changed:\n"
        #         msg += param_changed
        #         logger.warning(msg)            

    # @abstractmethod
    # def set_init_params(self) -> None:
    #     """Initialize the model's trainable parameters."""

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

    def model_args(self) -> Dict[str, Any]:
        """Retrieve model arguments.

        Returns:
            Model arguments
        """
        return self._model.model_args

    def training_args(self) -> Dict[str, Any]:
        """Retrieve training arguments.

        Returns:
            Training arguments
        """
        return self._training_args

    # def get_learning_rate(self, lr_key: str = 'eta0') -> List[float]:
    #     lr = self._model.model_args.get(lr_key)
    #     if lr is None:
    #         # get the default value
    #         lr = self._model.__dict__.get(lr_key)
    #     if lr is None:
    #         raise FedbiomedTrainingPlanError("Cannot retrieve learning rate. As a quick fix, specify it in the Model_args")
    #     return [lr]

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
        # Message to format for unexpected argument definitions in special methods
        method_error = \
            ErrorNumbers.FB605.value + ": Special method `{method}` has more than one argument: {keys}. This method " \
                                       "can not have more than one argument/parameter (for {prefix} arguments) or " \
                                       "method can be defined without argument and `{alternative}` can be used for " \
                                       "accessing {prefix} arguments defined in the experiment."
        
        if self._model is None:
            raise FedbiomedTrainingPlanError("can not configure optimizer, Model is None")
        # Get optimizer defined by researcher ---------------------------------------------------------------------
        init_optim_spec = get_method_spec(self.init_optimizer)
        if not init_optim_spec:
            optimizer = self.init_optimizer()
        elif len(init_optim_spec.keys()) == 1:
            optimizer = self.init_optimizer(self._optimizer_args)
        else:
            raise FedbiomedTrainingPlanError(method_error.format(prefix="optimizer",
                                                                 method="init_optimizer",
                                                                 keys=list(init_optim_spec.keys()),
                                                                 alternative="self.optimizer_args()"))
        # create optimizer builder
        optim_builder = OptimizerBuilder()
        
        # then build optimizer wrapper given model and optimizer
        self._optimizer = optim_builder.build(self.__type, self._model, optimizer) 
        
        # if optimizer is None:
        #     # default case: no optimizer is passed, using native sklearn optimizer
        #     logger.debug("Using native sklearn optimizer")
        #     self._optimizer = NativeSkLearnOptimizer(self._model)
        # elif isinstance(optimizer, (DeclearnOptimizer, FedOptimizer)):
        #     logger.debug("using a declearn Optimizer")
        #     if isinstance(optimizer, FedOptimizer):
        #         optimizer = FedOptimizer(optimizer)
        #     self._optimizer = SkLearnOptimizer(self._model, optimizer)
            
        # else:
        #     raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Optimizer should be a declearn optimizer, but got {type(optimizer)}. If you want to use only native scikit learn optimizer, please do not define a `init_optimizer` method in the TrainingPlan")

    def init_optimizer(self) -> None:
        """Default optimizer, which basically returns None (meaning native inner scikit learn optimization will be used)"""
        pass

    def optimizer_args(self) -> Dict:
        """Retrieves optimizer arguments

        Returns:
            Optimizer arguments
        """
        return self._optimizer_args

    def get_model_params(self) -> Dict:
        return self.after_training_params()

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

        # if not isinstance(self.model(), BaseEstimator):
        #     msg = (
        #         f"{ErrorNumbers.FB320.value}: model should be a scikit-learn "
        #         f"estimator, but is of type {type(self.model())}"
        #     )
        #     logger.critical(msg)
        #     raise FedbiomedTrainingPlanError(msg)
        if not isinstance(self.training_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot "
                "be trained without a NPDataLoader as `training_data_loader`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Run preprocessing operations.
        self._preprocess()
        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):
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

    def save(
            self,
            filename: str,
            params: Union[None, Dict[str, np.ndarray], Dict[str, Any]] = None
        ) -> None:
        """Save the wrapped model and its trainable parameters.

        This method is designed for parameter communication. It
        uses the joblib.dump function, which in turn uses pickle
        to serialize the model. Note that unpickling objects can
        lead to arbitrary code execution; hence use with care.

        Args:
            filename: Path to the output file.
            params: Model parameters to enforce and save.
                This may either be a {name: array} parameters dict, or a
                nested dict that stores such a parameters dict under the
                'model_params' key (in the context of the Round class).

        Notes:
            Save can be called from Job or Round.
            * From [`Round`][fedbiomed.node.round.Round] it is called with params (as a complex dict).
            * From [`Job`][fedbiomed.researcher.job.Job] it is called with no params in constructor, and
                with params in update_parameters.
        """
        # Optionally overwrite the wrapped model's weights.
        if params:
            if isinstance(params.get('model_params'), dict):  # in a Round
                params = params["model_params"]
            # for key, val in params.items():
            #     setattr(self._model, key, val)
            self._model.set_weights(params)
        # Save the wrapped model (using joblib, hence pickle).
        self._model.save(filename)

    def load(
            self,
            filename: str,
            to_params: bool = False
        ) -> Union[BaseEstimator, Dict[str, Dict[str, np.ndarray]]]:
        """Load a scikit-learn model dump, overwriting the wrapped model.

        This method uses the joblib.load function, which in turn uses
        pickle to deserialize the model. Note that unpickling objects
        can lead to arbitrary code execution; hence use with care.

        This function updates the `_model.model` private attribute with the
        loaded instance, and returns either that same model or a dict
        wrapping its trainable parameters.

        Args:
            filename: The path to the pickle file to load.
            to_params: Whether to return the model's parameters
                wrapped as a dict rather than the model instance.

        Notes:
            Load can be called from a Job or Round:
            * From [`Round`][fedbiomed.node.round.Round] it is called to return the model.
            * From [`Job`][fedbiomed.researcher.job.Job] it is called with to return its parameters dict.

        Returns:
            Dictionary with the loaded parameters.
        """
        # Deserialize the dump, type-check the instance and assign it.
        self._model.load(filename)
        if not isinstance(self.model(), self._model_cls):
            msg = (
                f"{ErrorNumbers.FB304.value}: reloaded model does not conform "
                f"to expectations: should be of type {self._model_cls}, not "
                f"{type(self.model())}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Optionally return the model's pseudo state dict instead of it.
        if to_params:
            params = self._model.get_weights()
            return {"model_params": params}
        return self.model()

    def type(self) -> TrainingPlans:
        """Getter for training plan type """
        return self.__type

    def after_training_params(self) -> Dict[str, np.ndarray]:
        """Return the wrapped model's trainable parameters' current values.

        This method returns a dict containing parameters that need
        to be reported back and aggregated in a federated learning
        setting.

        Returns:
            dict[str, np.ndarray]: the trained parameters to aggregate.
        """
        #return {key: getattr(self._model, key) for key in self._param_list}
        return self._model.get_weights()
