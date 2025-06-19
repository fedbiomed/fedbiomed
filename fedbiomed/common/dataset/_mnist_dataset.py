# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for MNIST
"""
from pathlib import Path
from typing import Tuple, Union

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
    def __init__(
        self,
        root: Union[str, Path],
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        # Do we want generic transforms for MNIST ?
        # generic_transform : Transform = None,
        # generic_target_transform : Transform = None,
    ) -> None:
        super().__init__(
            root=root,
            framework_transform=framework_transform,
            framework_target_transform=framework_target_transform,
        )

    def __len__(self) -> int:
        """Get number of samples"""
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""
        data, target = self._get_nontransformed_item(index=index)

        if self._to_format == DataReturnFormat.DEFAULT:
            data_item = {
                "data": DatasetDataItemModality(
                    modality_name="data",
                    type=DataType.IMAGE,
                    data=data.numpy(),  # CPU tensor not tracked by autograd
                )
            }
            target_item = {"target": int(target)}
        elif self._to_format == DataReturnFormat.TORCH:
            data_item = {"data": data}
            target_item = {"target": target}
        else:
            raise FedbiomedError(
                "DataReturnFormat not supported by __getitem__ in MnistDataset"
            )

        return data_item, target_item
