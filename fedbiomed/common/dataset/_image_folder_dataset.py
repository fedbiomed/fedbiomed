# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for Image Folder
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from pandas import DataFrame

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
                    data=DataFrame([target["target"]]),
                )
            }

        elif self._to_format == DataReturnFormat.TORCH:
            # values and shape do not change between PIL Image, numpy and torch
            data_item = {"data": torch.from_numpy(np.array(data["data"]))}
            target_item = {"target": torch.tensor(target["target"])}

        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: DataReturnFormat not supported"
            )

        return data_item, target_item

    def to_torch(self) -> None:
        self._to_format = DataReturnFormat.TORCH
