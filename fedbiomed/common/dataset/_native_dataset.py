# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import (
    Dataset as TorchDataset,
)

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError


class NativeDataset(Dataset):
    _native_to_framework = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: lambda x: (
            T.ToTensor()(x)  # In case the target is a PIL Image
            if isinstance(x, (Image.Image, np.ndarray))
            else torch.tensor(x)
        ),
    }

    def __init__(self, dataset, target=None):
        """Initialize the dataset with necessary checks.

        Args:
            dataset: the dataset (could be a TorchDataset or others)
            target: the target labels or data to be used

        Raises:
            Fedbiomed error if the len or getitem methods are not implemented,
            or if it fails to fetch a sample.
        """

        # Check if len and __getitem__ are implemented in the dataset
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise FedbiomedError(
                ErrorNumbers.FB632
                + " Dataset must implement __len__ and __getitem__ methods"
            )

        # Type checking if dataset is a TorchDataset or not
        self.is_torch_dataset = isinstance(dataset, TorchDataset)

        # Check if you can get a sample from the dataset
        try:
            sample = dataset[0]  # Try to fetch the first sample
        except Exception as e:
            raise FedbiomedError(
                ErrorNumbers.FB632
                + "Failed to get an item from dataset. Detailed error message: "
                + str(e)
            ) from e

        # Check if the sample returns a tuple or not
        # Initialize data and target accordingly
        if isinstance(sample, tuple):
            self.is_tuple_dataset = True
            self.data, self.target = [], []
            for x, t in dataset:  # each item: (PIL/torch/numpy, label)
                self.data.append(
                    torch.as_tensor(np.array(x))
                    if not isinstance(x, torch.Tensor)
                    else x
                )
                self.target.append(int(t))
        else:
            self.is_supervised = False
            self.data = dataset
            self.target = target

        if self.is_tuple_dataset and target is not None:
            # alitolga: TODO: We can raise an error and stop the execution as well.
            # There is a design choice to be made here.
            print(
                "WARNING: Target found both in dataset and target. "
                "Progressing with the target parameter."
            )
            self.target = target  # Undo the previous initialization

    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated with expected return format
        """

        # alitolga: TODO: Do we want to try to implement Transforms as well?

        # Store the to_format for later use
        self.to_format = to_format

        # Ensure that the dataset returns the correct format (Torch or Sklearn)
        if to_format == DataReturnFormat.TORCH:
            # Apply the transformation to convert data to Torch tensors
            self.convert = self._native_to_framework[DataReturnFormat.TORCH]
        elif to_format == DataReturnFormat.SKLEARN:
            # Apply the transformation to convert data to numpy arrays (sklearn-compatible)
            self.convert = self._native_to_framework[DataReturnFormat.SKLEARN]
        else:
            raise ValueError(f"Unsupported return format: {to_format}")

        converted_data, converted_target = [], []
        for d, t in zip(self.data, self.target, strict=True):
            converted_data.append(self.convert(d))
            converted_target.append(self.convert(t))

        self.data = converted_data
        self.target = converted_target

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        """Returns a tuple of (data, target) with the specified index"""
        return self.data[idx], self.target[idx]

    def __len__(self) -> int:
        """Returns the length of the data"""
        return len(self.data)
