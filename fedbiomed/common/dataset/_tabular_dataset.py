# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_controller._tabular_controller import TabularController
from fedbiomed.common.dataset_types import DataReturnFormat, RowSpec
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


def _polars_to_torch(x: pl.DataFrame) -> "torch.Tensor":
    try:
        return x.to_torch().reshape(-1)
    except TypeError as e:
        non_numeric = [
            name for name, dtype in x.schema.items() if not dtype.is_numeric()
        ]
        if non_numeric:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Column(s) {non_numeric} have non-numeric "
                "dtype and cannot be converted to a torch Tensor. Encode or drop them "
                "before creating the dataset."
            ) from e
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Failed to convert data to a torch Tensor: {e}"
        ) from e


class TabularDataset(Dataset):
    _controller_cls: type = TabularController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.to_numpy().reshape(-1),
        DataReturnFormat.TORCH: _polars_to_torch,
    }

    def __init__(
        self,
        input_columns: Iterable | int | str,
        target_columns: Optional[Iterable | int | str] = None,
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

        # Validation of columns is deferred to load
        # as self._controller._reader implements the logic to validate columns
        self._input_columns = input_columns
        self._target_columns = target_columns

    def load(
        self,
        to_format: DataReturnFormat,
        **controller_kwargs: Any,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            to_format: format associated to expected return format
            controller_kwargs: arguments to create controller
        """
        self.to_format = to_format

        self._init_controller(**controller_kwargs)

        # Normalize columns using controller (implies validation)
        self._input_columns = self._controller.normalize_columns(self._input_columns)
        if self._target_columns is not None:
            self._target_columns = self._controller.normalize_columns(
                self._target_columns
            )
            # Check for overlap between input_columns and target_columns
            _intersection_cols = list(
                set(self._input_columns) & set(self._target_columns)
            )
            if _intersection_cols:
                logger.warning(
                    f"Columns {_intersection_cols} are present in both input_columns and target_columns."
                )

        sample = self._controller.get_sample(0)  # type: ignore

        n_rows, _ = sample.shape
        if n_rows > 1:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: TabularDataset currently only supports "
                "row-wise samples. Sample obtained from controller has multiple rows."
            )

        self._validate_format_and_transformations(
            self._get_item_from_sample(sample, self._input_columns),
            transform=self._transform,
        )
        if self._target_columns is not None:
            self._validate_format_and_transformations(
                self._get_item_from_sample(sample, self._target_columns),
                transform=self._target_transform,
            )

    def _get_item_from_sample(
        self, sample: pl.DataFrame, columns: Optional[Iterable | int | str]
    ) -> Optional[pl.DataFrame]:
        """Get item from sample based on columns

        Args:
            sample: Data sample
            columns: Columns to select

        Returns:
            Selected columns or None if columns is None

        Raises:
            FedbiomedError: if any selected column has a non-numeric dtype
        """
        if columns is None:
            return None
        data = sample.select(columns)
        non_numeric = [
            name for name, dtype in data.schema.items() if not dtype.is_numeric()
        ]
        if non_numeric:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Column(s) {non_numeric} have non-numeric "
                "dtype and cannot be converted for training. Encode or drop them before "
                "creating the dataset."
            )
        return data

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Union["np.array", "torch.Tensor"], Optional[Union["np.array", "torch.Tensor"]]
    ]:
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
            "data": self._get_item_from_sample(sample, self._input_columns),
            "target": self._get_item_from_sample(sample, self._target_columns),
        }

        sample = self.apply_transforms(sample)
        return sample["data"], sample["target"]

    def analytics_schema(self):
        """Return schema for federated analytics"""
        return RowSpec(columns=self._input_columns), None
