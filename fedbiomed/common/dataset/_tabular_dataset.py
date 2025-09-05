from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union

if TYPE_CHECKING:
    import numpy as np
    import torch

import polars as pl

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_controller._tabular_controller import TabularController
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError


class TabularDataset(Dataset):
    _controller_cls: type = TabularController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.to_numpy(),
        DataReturnFormat.TORCH: lambda x: x.to_torch(),
    }

    def __init__(
        self,
        input_columns: Iterable | int | str,
        target_columns: Iterable | int | str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Constructor of the class

        Args:
            input_columns: Columns to be used as input features
            target_columns: Columns to be used as target
            transform: Transformation to be applied to input features
            target_transform: Transformation to be applied to target
        Raises:
            FedbiomedValueError: if `input_columns` or `target_columns` are not valid
            FedbiomedValueError: if `transform` or `target_transform` are not valid callables
        """

        # Transformation checks
        self._transform = self._validate_transform(transform=transform)
        self._target_transform = self._validate_transform(transform=target_transform)

        self._input_columns = input_columns
        self._target_columns = target_columns

    def complete_initialization(  # type: ignore
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated to expected return format
        """
        self.to_format = to_format

        self._init_controller(controller_kwargs=controller_kwargs)

        sample = self._controller.get_sample(0)  # type: ignore
        self._validate_transformations(
            self._get_inputs_from_sample(sample), transform=self._transform
        )
        self._validate_transformations(
            self._get_targets_from_sample(sample), transform=self._transform
        )

    def _get_inputs_from_sample(self, sample: pl.DataFrame) -> pl.DataFrame:
        """Get inputs dataset

        Returns:
            Input data
        """
        return sample.select(self._controller.normalize_columns(self._input_columns))

    def _get_targets_from_sample(self, sample: pl.DataFrame) -> pl.DataFrame:
        """Get target columns

        Returns:
            List of target columns
        """
        return sample.select(self._controller.normalize_columns(self._target_columns))

    def __getitem__(self, idx: int) -> Dict[str, Union["np.array", "torch.Tensor"]]:
        """Retrieve item at index `idx`

        Args:
            idx: index of the item to retrieve

        Raises:
            FedbiomedError: if dataset has not completed initialization

        Returns:
            A dictionary with keys "data" and "target" containing the respective items
        """
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: This dataset object has not completed initialization."
            )

        sample = self._controller.get_sample(idx)  # type: ignore

        sample = {
            "data": self._get_inputs_from_sample(sample),
            "target": self._get_targets_from_sample(sample),
        }

        sample = self.apply_transforms(sample)

        return {"data": sample["data"], "target": sample["target"]}
