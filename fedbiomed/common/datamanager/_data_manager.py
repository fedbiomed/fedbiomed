# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data Management factory class
"""

from typing import Any, Callable, Dict, Optional, Union

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset._native_dataset import NativeDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError

from ._framework_data_manager import FrameworkDataManager
from ._sklearn_data_manager import SkLearnDataManager
from ._torch_data_manager import TorchDataManager

_tp_to_datamanager: dict[TrainingPlans, type[FrameworkDataManager]] = {
    TrainingPlans.TorchTrainingPlan: TorchDataManager,
    TrainingPlans.SkLearnTrainingPlan: SkLearnDataManager,
}

_dm_to_format: dict[Callable, DataReturnFormat] = {
    TorchDataManager: DataReturnFormat.TORCH,
    SkLearnDataManager: DataReturnFormat.SKLEARN,
}


class DataManager(object):
    """Factory class that builds different data loaders

    Data loader type is based on the framework of the training plan.

    If `dataset` is not yet a `Dataset`, it also wraps it in a `NativeDataset` object.
    """

    _loader_arguments: Dict
    _dataset: Dataset
    _data_manager_instance: Optional[FrameworkDataManager] = None

    def __init__(
        self, dataset: Union[Dataset, Any], target: Optional[Any] = None, **kwargs: dict
    ) -> None:
        """Constructor of DataManager,

        Args:
            dataset: Either an already structured `Dataset` or the data component of
                unformatted dataset
            target: Target component of unformatted dataset, or `None` for an already
                structured dataset
            **kwargs: Additional parameters that are going to be used for data loader

        Raises:
            FedbiomedError: using targets with structured dataset
            FedbiomedError: cannot create a native dataset from unformatted data
        """
        # no type check needed, kwargs are dict
        self._loader_arguments = kwargs

        if isinstance(dataset, Dataset):
            if target is not None:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: cannot use `target` argument "
                    f"when using a formatted dataset. Targets are already part of the "
                    f"`Dataset` argument"
                )
        else:
            try:
                dataset = NativeDataset(dataset, target)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}:"
                    f" Init DataManager fails to initialize Native Dataset"
                    f" with error message {str(e)}"
                ) from e

        self._dataset = dataset

    def extend_loader_args(self, extension: Optional[Dict]) -> None:
        """Extends the class' loader arguments

        Extends the class's `_loader_arguments` attribute with additional key-values from
        the `extension` argument. If a key already exists in the `_loader_arguments`, then
        it is not replaced.

        Args:
            extension: the mapping used to extend the loader arguments
        """
        if extension:
            self._loader_arguments.update(
                {
                    key: value
                    for key, value in extension.items()
                    if key not in self._loader_arguments
                }
            )

    def load(self, tp_type: TrainingPlans) -> None:
        """Loads proper DataManager based on given TrainingPlan and
        `dataset`, `target` attributes.

        Args:
            tp_type: Enumeration instance of TrainingPlans that stands for type of training plan.

        Raises:
            FedbiomedError: unknown training plan type

        """
        if tp_type in _tp_to_datamanager:
            self._data_manager_instance = _tp_to_datamanager[tp_type](
                dataset=self._dataset, **self._loader_arguments
            )
        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unknown training plan type, "
                "cannot instantiate data manager."
            )

    def complete_dataset_initialization(
        self, controller_kwargs: Dict[str, Any]
    ) -> None:
        """Finalizes initialization of the DataManager's dataset controller

        Args:
            controller_kwargs: arguments for the controller

        Raises:
            FedbiomedError: if `_data_manager_instance` is not initialized
            FedbiomedError: if there is a problem completing dataset initialization
        """
        if not self._data_manager_instance:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Data manager instance is not initialized. "
                f"Please call `load()` first."
            )

        try:
            self._dataset.complete_initialization(
                controller_kwargs,
                _dm_to_format[self._data_manager_instance.__class__],
            )
        except FedbiomedError as e:
            raise e
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to complete dataset initialization."
            ) from e

    def __getattr__(self, item: str) -> Any:
        """Wraps all functions/attributes of factory class members.

        Args:
            item: Requested item from class

        Returns:
            Depends on the item

        Raises:
            FedbiomedError: the attribute is not implemented

        """

        # Specific to DataManager class
        if item == "load":
            return object.__getattribute__(self, item)

        try:
            return self._data_manager_instance.__getattribute__(item)
        except AttributeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: method {str(item)} not "
                f"implemented for class: {str(self._data_manager_instance)}"
            ) from e
