from typing import List
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.utils import get_class_source


class BaseTrainingPlan(object):
    def __init__(self):
        """ A Base class that includes common method that are used for
        all training plans

        Attrs:
            dependencies (List): All the dependencies that are need to be import
                                TrainingPlan as module
            dataset_path (string): The path that indicates where dataset has been stored
        """

        super(BaseTrainingPlan, self).__init__()
        self.dependencies = []
        self.dataset_path = None

    def add_dependency(self, dep: List[str]):
        """ Add snew dependency to the TrainingPlan class. These dependencies are used
        while creating a python module.

        Args:
           dep (List[string]): Dependency to add. Dependencies should be indicated as import string
                                e.g. `from torch import nn`
        """

        self.dependencies.extend(dep)

    def set_dataset(self, dataset_path):
        """ Dataset path setter for TrainingPlan

        Args:
            dataset_path (str): The path where data is saved on the node. This method is called by
                                the node who will execute the training.

        """
        self.dataset_path = dataset_path
        logger.debug('Dataset path has been set as' + self.dataset_path)

    def save_code(self, filepath: str):
        """Save the class code for this training plan to a file

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
            raise FedbiomedTrainingPlanError(ErrorNumbers.FB605.value + f"Error while getting source of the "
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
            _msg = ErrorNumbers.FB605.value + f" : Unable to read {filepath} due to unsatisfactory privileges"
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

        # Return filepath and content this allows
        return filepath, content
