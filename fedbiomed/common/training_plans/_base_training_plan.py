"""
A Base class that includes common methods that are used for
all training plans
"""

from collections import OrderedDict
from typing import List, Callable

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_class_source
from fedbiomed.common.constants import ProcessTypes


class BaseTrainingPlan(object):
    def __init__(self):
        """
        Base constructor

        Attrs:
            dependencies (list): All the dependencies that are need to be imported to create TrainingPlan as module.
                                 Dependencies are `import` statements as string. e.g. `"import numpy as np"`
            dataset_path (string): The path that indicates where dataset has been stored
        """

        super().__init__()
        self.dependencies = []
        self.dataset_path = None
        self.pre_processes = OrderedDict()

    def add_dependency(self, dep: List[str]):
        """
        Adds new dependency to the TrainingPlan class. These dependencies are used
        while creating a python module.

        Args:
           dep (List[string]): Dependency to add. Dependencies should be indicated as import string
                                e.g. `from torch import nn`
        """

        self.dependencies.extend(dep)

    def set_dataset_path(self, dataset_path):
        """
        Dataset path setter for TrainingPlan

        Args:
            dataset_path (str): The path where data is saved on the node. This method is called by
                                the node who will execute the training.

        """
        self.dataset_path = dataset_path
        logger.debug('Dataset path has been set as' + self.dataset_path)

    def save_code(self, filepath: str):
        """
        Saves the class source/codes of the training plan class that is created by
        user.

        Args:
            filepath (string): path to the destination file

        Returns:
            None

        Exceptions:
            FedBioMedTrainingPlanError:
        """

        try:
            class_source = get_class_source(self.__class__)
        except FedbiomedError as e:
            raise FedbiomedTrainingPlanError(ErrorNumbers.FB605.value +
                                             ": Error while getting source of the " +
                                             f"model class - {e}")

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
            logger.error(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except MemoryError:
            _msg = ErrorNumbers.FB605.value + f" : Can't write model file on {filepath}: out of memory!"
            logger.error(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB605.value + f" : Can't open file {filepath} to write model content"
            logger.error(_msg)
            raise FedbiomedTrainingPlanError(_msg)

        # Return filepath and content
        return filepath, content

    def training_data(self):
        """
        all subclasses must provide a training_data routine
        the purpose of this actual code is to detect that it has been provided

        :raise FedbiomedTrainingPlanError if called
        """
        msg = ErrorNumbers.FB303.value + ": training_data must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def add_preprocess(self, method: Callable, process_type: ProcessTypes):
        """
        Method adding preprocesses
        """
        if not isinstance(method, Callable):
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Error while adding preprocess, preprocess "
                                             f"should be a callable method")

        if not isinstance(process_type, ProcessTypes):
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Error while adding preprocess, process type "
                                             f"should be an instance of `fedbiomed.common.constants.ProcessType`")

        self.pre_processes[method.__name__] = {
            'method': method,
            'process_type': process_type
        }
