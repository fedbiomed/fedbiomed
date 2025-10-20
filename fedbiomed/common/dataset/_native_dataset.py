# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError


class NativeDataset(Dataset):
    """A class representing a native dataset.

    This class wraps around datasets from popular ML libraries like PyTorch and
    scikit-learn, allowing them to be used seamlessly in a customized TrainingPlan for FedBiomed.
    """

    _native_to_framework = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: lambda x: (
            T.ToTensor()(x)  # In case the input is a PIL image or ndarray image
            if isinstance(x, (Image.Image, np.ndarray))
            else torch.tensor(x)
        ),
    }

    def __init__(self, dataset, target: Optional[Any] = None):
        """Initialize with basic checks, without loading data to memory.

        Args:
            dataset: Native dataset object from a ML library (e.g., PyTorch, scikit-learn).
            target: Optional target data if not included in the dataset.
        Raises:
            FedbiomedError: if dataset does not implement collection interface,
                or if target length does not match dataset length,
                or if both dataset and argument provide targets.
        """
        # Check collection interface
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset must implement __len__ and __getitem__."
            )

        self._dataset = dataset

        # Probe one sample to determine supervised/unsupervised shape
        try:
            sample = dataset[0]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to get a sample item from dataset. Details: {e}"
            ) from e

        self._is_supervised = isinstance(sample, tuple)

        # If both dataset and argument provide targets -> conflict
        if self._is_supervised and target is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Target found both in dataset and in 'target' argument."
            )

        # Raise an error if length of target does not match dataset length
        if hasattr(target, "__len__") and len(target) != len(dataset):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Length of target ({len(target)}) does not match dataset ({len(dataset)})."
            )

        self._target = target  # may be None
        self._to_format: Optional[DataReturnFormat] = None

    def complete_initialization(
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat,
    ) -> None:
        """Select data and target, and check if they can be converted to requested format.

        Args:
            controller_kwargs: keyword arguments for controller (not used here).
            to_format: format associated to expected return format.
        Raises:
            FedbiomedError: if there is a problem converting dataset items to requested format.
        """

        self._to_format = to_format
        self._converter = self._get_format_conversion_callable()

        if self._is_supervised:
            data, target = self._dataset[0]
        elif self._target is not None:
            data = self._dataset[0]
            target = self._target[0]
        else:
            data = self._dataset[0]
            target = None

        try:
            self._validate_format_conversion(data)
        except FedbiomedError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to convert dataset items to "
                f"requested format {to_format}. Details: {e}"
            ) from e

        if target is not None:
            try:
                self._validate_format_conversion(target)
            except FedbiomedError as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to convert dataset items to "
                    f"requested format {to_format}. Details: {e}"
                ) from e

    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Fetch one item and convert to requested framework format.

        Args:
            idx: index of the item to fetch.
        Returns:
            A tuple (data, target) converted to the requested format.
        Raises:
            FedbiomedError: if there is a problem converting data or target to requested format.
        """
        if self._is_supervised:
            data, target = self._dataset[idx]
        else:
            data = self._dataset[idx]
            target = self._target[idx] if self._target is not None else None

        # Convert on-the-fly
        try:
            data_cvt = self._converter(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to convert data item at index {idx}. Details: {e}"
            ) from e
        try:
            target_cvt = self._converter(target) if target is not None else None
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to convert target item at index {idx}. Details: {e}"
            ) from e

        return data_cvt, target_cvt

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self._dataset)
