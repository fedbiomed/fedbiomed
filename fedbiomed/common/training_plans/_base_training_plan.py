"""Base class defining the shared API of all training plans."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers, ProcessTypes
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_class_source


class BaseTrainingPlan(metaclass=ABCMeta):
    """Base class for training plan

    All concrete, framework- and/or model-specific trainning plans should
    inherit from this class, implement its `post_init` abstract method,
    and more generally extend (or overwrite) its API-defining methods.

    Attrs:
        dependencies: All the dependencies that need to be imported
          to create the TrainingPlan on nodes' side. Dependencies
          are import statements strings, e.g. `"import numpy as np"`.
        dataset_path: The path that indicates where dataset has been stored
        pre_process: Preprocess functions that will be applied to the training
          data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    def __init__(self) -> None:
        """Construct the base training plan."""
        self._dependencies = []  # type: List[str]
        self.dataset_path = None  # type: Union[str, None]
        self.pre_processes = OrderedDict()
        self.training_data_loader = None  # type: Union[DataLoader, NPDataLoader, None]
        self.testing_data_loader = None  # type: Union[DataLoader, NPDataLoader, None]

    @abstractmethod
    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
        ) -> None:
        """Set arguments for the model, training and the optimizer.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
        """
        return None

    def add_dependency(self, dep: List[str]) -> None:
        """Add new dependencies to the TrainingPlan.

        These dependencies are used while creating a python module.

        Args:
            dep: Dependencies to add. Dependencies should be indicated as
                import statement strings, e.g. `"from torch import nn"`.
        """
        for val in dep:
            if val not in self._dependencies:
                self._dependencies.append(val)

    def set_dataset_path(self, dataset_path: str) -> None:
        """Dataset path setter for TrainingPlan

        Args:
            dataset_path (str): The path where data is saved on the node. This method is called by the node
                who will execute the training.
        """
        self.dataset_path = dataset_path
        logger.debug('Dataset path has been set as' + self.dataset_path)

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
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def save_code(self, filepath: str):
        """Saves the class source/codes of the training plan class that is created byuser.

        Args:
            filepath (string): path to the destination file

        Raises:
            FedBioMedTrainingPlanError: raised when source of the model class cannot be assessed
            FedBioMedTrainingPlanError: raised when model file cannot be created/opened/edited
        """
        try:
            class_source = get_class_source(self.__class__)
        except FedbiomedError as e:
            msg = ErrorNumbers.FB605.value + \
                  " : error while getting source of the model class - " + \
                  str(e)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Preparing content of the module
        content = ""
        for s in self._dependencies:
            content += s + "\n"

        content += "\n"
        content += class_source

        try:
            # should we write it in binary (for the sake of space optimization)?
            file = open(filepath, "w")
            file.write(content)
            file.close()
            logger.debug("Model file has been saved: " + filepath)
        except PermissionError:
            _msg = ErrorNumbers.FB605.value + f" : Unable to read {filepath} due to unsatisfactory privileges" + \
                   ", can't write the model content into it"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except MemoryError:
            _msg = ErrorNumbers.FB605.value + f" : Can't write model file on {filepath}: out of memory!"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB605.value + f" : Can't open file {filepath} to write model content"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)

        # Return filepath and content
        return filepath, content

    def training_data(self):
        """All subclasses must provide a training_data routine the purpose of this actual code is to detect
        that it has been provided

        Raises:
            FedbiomedTrainingPlanError: if called and not inherited
        """
        msg = ErrorNumbers.FB303.value + ": training_data must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def add_preprocess(
            self,
            method: Callable,
            process_type: ProcessTypes
        ) -> None:
        """Register a pre-processing method to be executed on training data.

        Args:
            method: Pre-processing method to be run before training.
            process_type: Type of pre-processing that will be run.
              The expected signature of `method` and the arguments
              passed to it depend on this parameter.
        """
        if not callable(method):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `method` should be callable."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if not isinstance(process_type, ProcessTypes):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `process_type` should be an instance "
                "of `fedbiomed.common.constants.ProcessType`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # NOTE: this may be revised into a list rather than OrderedDict
        self.pre_processes[method.__name__] = {
            'method': method,
            'process_type': process_type
        }

    def _preprocess(self) -> None:
        """Executes registered data pre-processors."""
        for name, process in self.pre_processes.items():
            method = process['method']
            process_type = process['process_type']
            if process_type == ProcessTypes.DATA_LOADER:
                self._process_data_loader(method=method)
            else:
                logger.error(
                    f"Process type `{process_type}` is not implemented."
                    f"Preprocessor '{name}' will therefore be ignored."
                )

    def _process_data_loader(
            self,
            method: Callable[..., Any]
        ) -> None:
        """Handle a data-loader pre-processing action.

        Args:
            method (Callable) : Process method that is to be executed.

        Raises:
            FedbiomedTrainingPlanError:
              - if the method does not have 1 positional argument (dataloader)
              - if running the method fails
              - if the method does not return a dataloader of the same type as
               its input
        """
        # Check that the preprocessing method has a proper signature.
        argspec = utils.get_method_spec(method)
        if len(argspec) != 1:
            msg = (
                f"{ErrorNumbers.FB605.value}: preprocess method of type "
                "`PreprocessType.DATA_LOADER` sould expect one argument: "
                "the data loader wrapping the training dataset."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Try running the preprocessor.
        try:
            data_loader = method(self.training_data_loader)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while running "
                f"preprocess method `{method.__name__}` -> {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        logger.debug(
            f"The process `{method.__name__}` has been successfully executed."
        )
        # Verify that the output is of proper type and assign it.
        if isinstance(data_loader, type(self.training_data_loader)):
            self.training_data_loader = data_loader
            logger.debug(
                "Data loader for training routine has been updated "
                f"by the process `{method.__name__}`."
            )
        else:
            msg = (
                f"{ErrorNumbers.FB605.value}: the return type of the "
                f"`{method.__name__}` preprocess method was expected "
                f"to be {type(self.training_data_loader)}, but was "
                f"{type(data_loader)}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    @staticmethod
    def _create_metric_result_dict(metric: Union[dict, list, int, float, np.ndarray, torch.Tensor, List[torch.Tensor]],
                                   metric_name: str = 'Custom') -> Dict[str, float]:
        """Create metric result dictionary.

        Args:
            metric: Array-like metric values or dictionary
            metric_name: Name of the metric. If `metric` is of type list, metric names will be in format of
                (`metric_name_1`, ..., `metric_name_n`), where `n` is the size of the list.
                If the `metric` argument is  provided as dict the argument `metric_name` will be ignored and
                metric names will be keys of the dict.

        Returns:
            Dictionary mapping <metric_name>:<metric values>, where <metric values> are floats provided by `metric`
                argument. If `metric` argument is a dict, then returns <keys of metric>: <metric values>

        Raises:
            FedbiomedTrainingPlanError: triggered if metric is not of type dict, list, int, float, torch.Tensor,
                or np.ndarray.
        """
        if isinstance(metric, torch.Tensor):
            metric = metric.numpy()
            metric = list(metric) if metric.shape else float(metric)
        elif isinstance(metric, np.ndarray):
            metric = list(metric)

        # If it is single int/float metric value
        if isinstance(metric, (int, float, np.integer, np.floating)) and not isinstance(metric, bool):
            return {metric_name: float(metric)}

        # If metric function returns multiple values
        elif isinstance(metric, list) or isinstance(metric, dict):

            if isinstance(metric, list):
                metric_names = [f"{metric_name}_{i + 1}" for i, val in enumerate(metric)]
            else:
                metric_names = metric.keys()

            try:
                metric = utils.convert_iterator_to_list_of_python_floats(metric)
            except FedbiomedError as e:
                msg = ErrorNumbers.FB605.value + \
                      " : wrong typed metric value - " + \
                      str(e)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            return dict(zip(metric_names, metric))

        else:
            msg = ErrorNumbers.FB605.value + \
                  " : metric value should be one of type" + \
                  " int, float, np.integer, torch.Tensor," + \
                  "list of int/float/np.integer/torch.Tensor or" + \
                  "dict of key (int/float/np.integer/torch.Tensor) instead of " + \
                  str(type(metric))
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
