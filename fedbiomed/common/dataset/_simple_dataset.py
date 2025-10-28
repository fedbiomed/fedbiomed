# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import (
    ImageFolderController,
    MedNistController,
    MnistController,
)
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset


class _SimpleDataset(Dataset):
    "Dataset where data and target are implicitly predefined by the controller"

    _native_to_framework = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: lambda x: (
            T.ToTensor()(x)  # In case the target is a PIL Image
            if isinstance(x, (Image.Image, np.ndarray))
            else torch.tensor(x).float()
        ),
    }

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        if type(self) is _SimpleDataset:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: "
                "`SimpleDataset` cannot be instantiated directly"
            )
        self._transform = self._validate_transform(transform)
        self._target_transform = self._validate_transform(target_transform)

    def complete_initialization(
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

        sample = self._controller.get_sample(0)
        self._validate_format_and_transformations(
            sample["data"],
            self._transform,
            extra_info="Error raised by 'data'",
        )
        self._validate_format_and_transformations(
            sample["target"],
            self._target_transform,
            extra_info="Error raised by 'target'",
        )

    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset object has not completed "
                "initialization. It is not ready to use yet."
            )

        sample = self._controller.get_sample(idx)
        sample = self.apply_transforms(sample)

        return sample["data"], sample["target"]


class ImageFolderDataset(_SimpleDataset):
    _controller_cls = ImageFolderController


class MedNistDataset(_SimpleDataset):
    _controller_cls = MedNistController


class MnistDataset(_SimpleDataset):
    _controller_cls = MnistController
