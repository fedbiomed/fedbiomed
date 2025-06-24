# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for MNIST
"""

from pathlib import Path
from typing import Tuple, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._mnist_controller import MnistController
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetDataItem,
    DatasetDataItemModality,
    DataType,
    Transform,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._dataset import StructuredDataset


class MnistDataset(StructuredDataset, MnistController):
    """Interface of MnistController to return data samples in specific format."""

    def __init__(
        self,
        root: Union[str, Path],
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
    ) -> None:
        """Constructor of the class"""
        super().__init__(
            root=root,
            framework_transform=framework_transform,
            framework_target_transform=framework_target_transform,
        )

    def __len__(self) -> int:
        """Get number of samples"""
        return len(self._data)

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
                    data=data["data"].numpy(),
                )
            }
            target_item = {"target": target["target"].numpy()}
            return data_item, target_item

        if self._to_format == DataReturnFormat.TORCH:
            return data, target

        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: DataReturnFormat not supported"
        )
