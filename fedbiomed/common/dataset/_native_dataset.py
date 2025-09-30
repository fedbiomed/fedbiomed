from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import (
    Dataset as TorchDataset,
)

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError


class NativeDataset:
    _native_to_framework = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: lambda x: (
            T.ToTensor()(x)  # In case the target is a PIL Image
            if isinstance(x, (Image.Image, np.ndarray))
            else torch.tensor(x)
        ),
    }

    """ Meeting Notes and ToDo:

    Init:
    
    - Type checking (if dataset a TorchDataset or not)
    - Check if target is null
    - Check if you can get a sample
    - Check if the sample returns a tuple (supervised) or not (unsupervised)

    Complete_Initialization:

    - Check if it does have transforms/conversions 
    - Assure that it returns the same format (Tensor or Numpy) as the Framework (to_format) (Torch or SkLearn)
    
    """

    def __init__(self, dataset, target):
        """Initialize the dataset with necessary checks

        Args:
            dataset: the dataset (could be a TorchDataset or others)
            target: the target labels or data to be used

        ToDo:
        - Type checking (if dataset a TorchDataset or not)
        - Check if target is null
        - Check if you can get a sample from the dataset
        - Check if the sample returns a tuple (supervised) or not (unsupervised)
        """

        # Check if len and __getitem__ are implemented in the dataset
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise FedbiomedError(
                ErrorNumbers.FB632
                + " Dataset must implement __len__ and __getitem__ methods"
            )

        # Type checking if dataset is a TorchDataset or not
        if isinstance(dataset, TorchDataset):
            self.is_torch_dataset = True
        else:
            self.is_torch_dataset = False

        # Check if target is null
        if target is not None:
            self.target = target

        self.dataset = dataset

        # Check if you can get a sample from the dataset
        try:
            sample = dataset[0]  # Try to fetch the first sample
        except Exception as e:
            raise FedbiomedError(
                ErrorNumbers.FB632
                + "Failed to get an item from dataset. Detailed error message: "
                + str(e)
            ) from e

        # Check if the sample returns a tuple (supervised) or not (unsupervised)
        if isinstance(sample, tuple):
            self.is_supervised = True
        else:
            self.is_supervised = False

        if self.is_supervised and self.target is not None:
            print(
                "WARNING: Target found both in dataset and target"
                "Progressing with the target parameter."
            )

        # Additional attributes
        self.transform = None  # Assume there might be transforms to be added
        self.target_transform = None  # Assume there might be transforms to be added
        self.to_format = None  # Placeholder for final format setting

    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated with expected return format
        """

        # Store the to_format for later use
        self.to_format = to_format

        # Check if the dataset has any transforms or conversions
        if hasattr(self.dataset, "transform") and self.dataset.transform is not None:
            self.transform = self.dataset.transform
            self.target_transform = self.dataset.transform

        if (
            hasattr(self.dataset, "target_transform")
            and self.dataset.target_transform is not None
        ):
            self.target_transform = self.dataset.target_transform

        # Ensure that the dataset returns the correct format (Torch or Sklearn)
        if to_format == DataReturnFormat.TORCH:
            # Apply the transformation to convert data to Torch tensors
            self.convert = self._native_to_framework[DataReturnFormat.TORCH]
        elif to_format == DataReturnFormat.SKLEARN:
            # Apply the transformation to convert data to numpy arrays (sklearn-compatible)
            self.convert = self._native_to_framework[DataReturnFormat.SKLEARN]
        else:
            raise ValueError(f"Unsupported return format: {to_format}")

        converted_dataset = []
        for sample in self.dataset:
            converted_sample = self.convert(sample)
            converted_dataset.append(converted_sample)

        self.dataset = converted_dataset
