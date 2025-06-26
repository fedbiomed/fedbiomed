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
    def __init__(
        self,
        root: Union[str, Path],
        is_mednist: bool = False,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
    ) -> None:
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
        """Retrieve a data sample"""
        # H, W = img.size; C = len(img.getbands());
        data, target = self._get_nontransformed_item(index=index)

        if self._to_format == DataReturnFormat.DEFAULT:
            # Default shape (H, W, C)
            data_item = {
                "data": DatasetDataItemModality(
                    modality_name="data",
                    type=DataType.IMAGE,
                    data=np.array(data["data"]),
                )
            }
            target_item = {"target": np.array(target["target"])}

        elif self._to_format == DataReturnFormat.TORCH:
            # Default prefered PyTorch shape (C, H, W)
            data_item = {"data": transforms.ToTensor()(data["data"])}
            target_item = {"target": torch.tensor(target["target"])}

        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: DataReturnFormat not supported"
            )

        return data_item, target_item
