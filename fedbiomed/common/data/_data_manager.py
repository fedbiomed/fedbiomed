"""
Data Management classes
"""

import copy
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from fedbiomed.common.constants import _BaseEnum, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDataManagerError

from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import NPDataLoader, SkLearnDataManager
from ._tabular_dataset import TabularDataset


class DataLoaderTypes(_BaseEnum):
    """Enum for data-loader classes."""

    NUMPY = NPDataLoader
    TORCH = DataLoader


class DataManagerTypes(_BaseEnum):
    """Enum for data-manager classes."""

    NUMPY = SkLearnDataManager
    TORCH = TorchDataManager


# Type hint aliases for Union of concrete classes under previous enums.
TypeDataLoader = Union[
    NPDataLoader,
    DataLoader,
]
TypeDataManager = Union[
    SkLearnDataManager,
    TorchDataManager,
]


class DataManager:
    """Factory class to build type-specific data managers."""

    ARRAY_TYPES = (pd.DataFrame, pd.Series, np.ndarray)

    def __init__(
        self,
        dataset: Union[np.ndarray, pd.DataFrame, pd.Series, Dataset],
        target: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
        **kwargs: Any
    ) -> None:
        """Instantiate a DataManager factory.

        Args:
            dataset: Either a PyTorch Dataset instance wrapping both inputs and
                targets, or an array-like data structure wrapping inputs.
            target: Optional array-like structure wrapping target labels.
                May be None if `dataset` is a PyTorch Dataset.
            **kwargs: Additional parameters that should be passed to the
                data-loader-type-specific manager built through this factory.
        """
        # TODO: Improve datamanager for auto loading by given dataset_path and other information
        # such as inputs variable indexes and target variables indexes
        self._dataset = dataset
        self._target = target
        self._loader_arguments = kwargs

    def get_loader_arguments(
            self,
        ) -> Dict[str, Any]:
        """Return data-loader keyword arguments managed by this factory."""
        return copy.deepcopy(self._loader_arguments)

    def set_loader_arguments(
            self,
            loader_args: Dict[str, Any],
            update: bool = True,
        ) -> None:
        """Update the data-loader keyword arguments managed by this factory.

        Args:
            loader_args: Keyword arguments that should be passed to the
                data-loader-type-specific manager built through this factory.
            update: Whether to update the previous dict of arguments (keeping
                arguments that are not part of `loader_args`), or replace it.
        """
        if update:
            self._loader_arguments.update(loader_args)
        else:
            self._loader_arguments = copy.deepcopy(loader_args)

    def build(
        self,
        loader_type: DataLoaderTypes
    ) -> TypeDataManager:
        """Return a DataManager based on a DataLoader type specification.

        Args:
            loader_type: Type of dataloader (torch DataLoader, NPDataLoader...)
                required by the TrainingPlan.

        Raises:
            FedbiomedDataManagerError: If `loader_type` is unsupported, or
                if the wrapped (`dataset`, `target`) objects are of unsuited
                type to build the requested DataManager instance.
        """
        # Case when the training plan requires TorchDataLoader loaders.
        if loader_type is DataLoaderTypes.TORCH:
            return self._build_torch_manager()
        # Case when the training plan requires NPDataLoader loaders.
        if loader_type is DataLoaderTypes.NUMPY:
            return self._build_numpy_manager()
        # Case when the training plan specifies an unsupported loader type.
        raise FedbiomedDataManagerError(
            f"{ErrorNumbers.FB607.value}: "
            f"Unsupported data loader type: {loader_type}"
        )

    def _build_torch_manager(self) -> TorchDataManager:
        """Build a TorchDataManager from this factory."""
        # Case when a PyTorch Dataset was provided.
        if isinstance(self._dataset, Dataset) and self._target is None:
            return TorchDataManager(
                dataset=self._dataset, **self._loader_arguments
            )
        # Case when a pair of array-like structures were provided.
        if (
            isinstance(self._dataset, self.ARRAY_TYPES)
            and isinstance(self._target, self.ARRAY_TYPES)
        ):
            torch_dataset = TabularDataset(
                inputs=self._dataset, target=self._target
            )
            return TorchDataManager(
                dataset=torch_dataset, **self._loader_arguments
            )
        # Case of unsupported (dataset, target) pair of types.
        raise FedbiomedDataManagerError(
            f"{ErrorNumbers.FB607.value}: Invalid arguments to set up "
            "a TorchDataManager: either provide the argument  `dataset` "
            "as a PyTorch Dataset instance, or provide `dataset` and "
            "`target` arguments as pd.DataFrame, pd.Series or np.ndarray "
            "instances."
        )

    def _build_numpy_manager(self) -> SkLearnDataManager:
        """Build a SkLearnDataManager from this factory."""
        # Case when a PyTorch Dataset was provided.
        if isinstance(self._dataset, Dataset) and self._target is None:
            torch_data_manager = TorchDataManager(dataset=self._dataset)
            try:
                return torch_data_manager.to_sklearn()
            except Exception as exc:
                raise FedbiomedDataManagerError(
                    f"{ErrorNumbers.FB607.value}: An error occurred while "
                    "trying to convert a torch.utils.data.Dataset to a numpy-"
                    f"based dataset: {exc}"
                ) from exc
        # Case when a pair of array-like structures were provided.
        if (
            isinstance(self._dataset, self.ARRAY_TYPES)
            and isinstance(self._target, self.ARRAY_TYPES)
        ):
            return SkLearnDataManager(
                inputs=self._dataset, target=self._target,
                **self._loader_arguments
            )
        # Case of unsupported (dataset, target) pair of types.
        raise FedbiomedDataManagerError(
            f"{ErrorNumbers.FB607.value}: Invalid arguments to set up "
            "a SkLearnDataManager: either provide the argument  `dataset` "
            "as a PyTorch Dataset instance, or provide `dataset` and "
            "`target` arguments as pd.DataFrame, pd.Series or np.ndarray "
            "instances."
        )
