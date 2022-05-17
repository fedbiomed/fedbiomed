""" Module contains Base class that includes common methods that are used for all training plans"""


import numpy as np
import torch

from collections import OrderedDict
from typing import Tuple, Dict, List, Callable, Union

from torch.utils.data import DataLoader

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers, ProcessTypes
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_class_source


class BaseTrainingPlan(object):
    """Base training plan that should be inhereted by all other training plan classes

    Attrs:
        dependencies: All the dependencies that are need to be imported to create TrainingPlan as module.
            Dependencies are `import` statements as string. e.g. `"import numpy as np"`
        dataset_path: The path that indicates where dataset has been stored
        pre_process: Preprocess method that will be applied before training loop
        training_data_loader: Data loader for training routine/loop
        testing_data_loader: Data loader for testing routine
    """
    def __init__(self):
        """Construct base training plan"""

        super().__init__()
        self.dependencies = []
        self.dataset_path = None
        self.pre_processes = OrderedDict()
        self.training_data_loader = None
        self.testing_data_loader = None

    def add_dependency(self, dep: List[str]):
        """ Adds new dependency to the TrainingPlan class.

        These dependencies are used while creating a python module.

        Args:
           dep: Dependency to add. Dependencies should be indicated as import string. e.g. `from torch import nn`
        """

        self.dependencies.extend(dep)

    def set_dataset_path(self, dataset_path):
        """Dataset path setter for TrainingPlan

        Args:
            dataset_path (str): The path where data is saved on the node. This method is called by the node
                who will execute the training.
        """
        self.dataset_path = dataset_path
        logger.debug('Dataset path has been set as' + self.dataset_path)

    def set_data_loaders(self,
                         train_data_loader: Union[DataLoader, Tuple[np.ndarray, np.ndarray], None],
                         test_data_loader: Union[DataLoader, Tuple[np.ndarray, np.ndarray], None]):
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for testing routine
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
        for s in self.dependencies:
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

    def add_preprocess(self, method: Callable, process_type: ProcessTypes):
        """Adds preprocesses

        Args:
            method: preprocess method to be run before training
            process_type: Type of the process that will be run
        """
        if not isinstance(method, Callable):
            msg = ErrorNumbers.FB605.value + \
                " : error while adding preprocess, " + \
                "preprocess should be a callable method"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        if not isinstance(process_type, ProcessTypes):
            msg = ErrorNumbers.FB605.value + \
                " : error while adding preprocess," + \
                " process type should be an instance of" + \
                " `fedbiomed.common.constants.ProcessType`"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self.pre_processes[method.__name__] = {
            'method': method,
            'process_type': process_type
        }

    @staticmethod
    def _create_metric_result_dict(metric: Union[dict, list, int, float, np.ndarray, torch.Tensor, List[torch.Tensor]],
                                   metric_name: str = 'Custom') -> Dict[str, float]:
        """Create metric result dictionary.

        Args:
            metric: Array-like metric values, dictionary or single-value
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

        try:
            metric = utils.convert_to_python_float(metric)
            return {metric_name: float(metric)}
        except FedbiomedError:
            # Means that `metric` is array-like or unsupported type of value, pass to next controller
            pass

        # If metric function returns multiple values
        if isinstance(metric, list) or isinstance(metric, dict):

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
                "list of int/float/np.integer/torch.Tensor or " + \
                "dict of key: value as type int/float/np.integer/torch.Tensor, but got " + \
                str(type(metric))
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
