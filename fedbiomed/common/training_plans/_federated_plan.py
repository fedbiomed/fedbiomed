# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
import importlib
import os
import sys
from abc import ABC
from typing import List, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_method_spec, get_class_source


class FederatedPlan(ABC):
    def __init__(self) -> None:
        """Construct the base training plan."""
        self._dependencies: List[str] = self.init_dependencies()

    def init_dependencies(self) -> List[str]:
        """Default method where dependencies are returned

        Returns:
            Empty list as default
        """
        return ["from fedbiomed.common.training_plans import FederatedDataPlan",
                "from fedbiomed.common.data import DataManager"]

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

    def configure_dependencies(self) -> None:
        """ Configures dependencies """
        init_dep_spec = get_method_spec(self.init_dependencies)
        if len(init_dep_spec.keys()) > 0:
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605}: `init_dependencies` should not take any argument. "
                f"Unexpected arguments: {list(init_dep_spec.keys())}"
            )
        dependencies = self.init_dependencies()
        if not isinstance(dependencies, (list, tuple)):
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605}: Expected dependencies are a list or "
                "tuple of str, but got {type(dependencies)}"
            )
        self.add_dependency(dependencies)

    def save_code(self, filepath: str) -> None:
        """Saves the class source/codes of the training plan class that is created by user.

        Args:
            filepath: path to the destination file

        Raises:
            FedbiomedTrainingPlanError: raised when source of the model class cannot be assessed
            FedbiomedTrainingPlanError: raised when model file cannot be created/opened/edited
        """
        try:
            class_source = get_class_source(self.__class__)
        except FedbiomedError as exc:
            msg = f"{ErrorNumbers.FB605.value}: error while getting source of the model class: {exc}"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

        # Preparing content of the module
        content = "\n".join(self._dependencies)
        content += "\n"
        content += class_source
        try:
            # should we write it in binary (for the sake of space optimization)?
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(content)
            logger.debug("Model file has been saved: " + filepath)
        except PermissionError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Unable to read {filepath} due to unsatisfactory privileges" + \
                   ", can't write the model content into it"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc
        except MemoryError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Can't write model file on {filepath}: out of memory!"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc
        except OSError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Can't open file {filepath} to write model content"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc

    @staticmethod
    def load_training_plan_from_file(
            module_file_path: Union[str, os.PathLike],
            training_plan_module: str,
            training_plan_name: str) -> 'FederatedPlan':
        """Import a training plan class from a file and create a training plan object instance.

        Args:
            module_file_path: the OS path to the directory containing the module file (often the experiment's tmp dir)
            training_plan_module: module name of the training plan file
            training_plan_name: the name of the training plan class

        Returns:
            The default-constructed training plan object
        """
        sys.path.insert(0, module_file_path)
        module = importlib.import_module(training_plan_module)
        train_class = getattr(module, training_plan_name)
        sys.path.pop(0)
        return train_class()
