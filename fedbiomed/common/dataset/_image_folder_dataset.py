# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for Image Folder
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torchvision import transforms

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._image_folder_controller import (
    ImageFolderController,
)
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetDataItem,
    DatasetDataItemModality,
    DataType,
    Transform,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._dataset import StructuredDataset


class ImageFolderDataset(StructuredDataset, ImageFolderController):
    """Interface of ImageFolderController to return data samples in specific format."""

    def __init__(
        self,
        root: Union[str, Path],
        is_mednist: bool = False,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
    ) -> None:
        """Constructor of the class.

        Args:
            root: Root directory path.
            is_mednist: If True, download MedNIST into indicated path and add Mednist
                to `self.root` if last folder in path is not called MedNIST.
            framework_transform: Functions to transform the input data.
            framework_target_transform: Functions to transform the target data.

        Raises:
            FedbiomedError:
            - if root is not valid or do not exist
            - if MedNIST download fails
            - if `datasets.ImageFolder` can not be initialized
                (classes, samples and loader)
            - if framework-transforms are not valid Transform types
        """
        super().__init__(
            root=root,
            is_mednist=is_mednist,
            framework_transform=framework_transform,
            framework_target_transform=framework_target_transform,
        )

    def __len__(self) -> int:
        """Get number of samples"""
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample in specific format

        Args:
            index (int): Index

        Raises:
            FedbiomedError: If data return format is not supported

        Returns:
            Tuple[DatasetDataItem, DatasetDataItem]: (data, target)
        """
        data, target = self._get_nontransformed_item(index=index)

        if self._to_format == DataReturnFormat.DEFAULT:
            # data_shape corresponds to (H, W, C) if C!=1 else (H, W)
            # e.g. PIL Image mode="L" size=(28, 28) equals np.ndarray of shape (28, 28)
            # PIL Image mode="RGB" size=(28, 28) equals np.ndarray of shape (28, 28, 3)
            data_item = {
                "data": DatasetDataItemModality(
                    modality_name="data",
                    type=DataType.IMAGE,
                    data=np.array(data["data"]),
                )
            }
            target_item = {
                "target": DatasetDataItemModality(
                    modality_name="target",
                    type=DataType.TABULAR,
                    data=np.array(target["target"]),
                )
            }

        elif self._to_format == DataReturnFormat.TORCH:
            # PIL Image or numpy.ndarray (H, W, C) in the range [0, 255] are transformed
            # into torch.FloatTensor of shape (C, H, W) in the range [0.0, 1.0]
            # e.g. PIL Image mode=L size=(28, 28) equals Tensor of shape (1, 28, 28)
            data_item = {"data": transforms.ToTensor()(data["data"])}
            target_item = {"target": torch.tensor(target["target"])}

        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: DataReturnFormat not supported"
            )

        return data_item, target_item

    def to_torch(self) -> None:
        self._to_format = DataReturnFormat.TORCH
