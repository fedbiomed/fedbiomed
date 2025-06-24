# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MNIST
"""

from pathlib import Path
from typing import Tuple, Union

from torchvision import datasets

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import (
    DatasetData,
    DatasetDataItem,
    DatasetDataModality,
    DatasetDataStructured,
    DataType,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._controller import Controller


class MnistController(Controller):
    """Generic Mnist controller where the data is arranged in this way:

        root
        └──MNIST
            └── raw
                ├── train-images-idx3-ubyte
                ├── train-labels-idx1-ubyte
                ├── t10k-images-idx3-ubyte
                └── t10k-labels-idx1-ubyte

    Uses `torchvision.datasets.MNIST`.
    """

    def __init__(self, root: Union[str, Path], **kwargs) -> None:
        """Constructor of the class.

        Automatically downloads and extracts the files if they do not exist.

        Data is directly loaded. Train files are used by default:
            - train-images-idx3-ubyte
            - train-labels-idx1-ubyte

        Args:
            root: Root directory path

        Raises:
            FedbiomedError:
            - if root is not valid or do not exist (from `Controller`)
            - if `datasets.MNIST` can not be initialized (data and targets)
        """
        try:
            self.root = root
            dataset = datasets.MNIST(root=self.root, download=True)
        except Exception as e:
            raise FedbiomedError(ErrorNumbers.FB632.value) from e

        self._data = dataset.data
        self._targets = dataset.targets
        self._dataset_data_meta = self._get_dataset_data_meta()
        super().__init__(**kwargs)

    def _get_nontransformed_item(
        self,
        index: int,
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Returns data and target associated to index.

        Args:
            index (int): Index

        Returns:
            Tuple[dict[str: torch.tensor]] (image: , target)
                where target tensor corresponds to index of the target class.
        """
        data_item = {"data": self._data[index]}
        target_item = {"target": self._targets[index]}
        return data_item, target_item

    def _get_dataset_data_meta(self) -> DatasetData:
        """Returns meta data of samples recovered with `_get_nontransformed_item`"""
        data_item, target_item = self._get_nontransformed_item(index=0)
        data_meta = {
            "data": DatasetDataModality(
                modality_name="data",
                type=DataType.IMAGE,
                shape=tuple(data_item["data"].shape),
            )
        }
        target_meta = {
            "target": DatasetDataModality(
                modality_name="target",
                type=DataType.TABULAR,
                shape=tuple(target_item["target"].shape),
            )
        }
        return DatasetDataStructured(
            data=data_meta,
            target=target_meta,
            len=len(self._data),
        )

    def validate(self):
        pass
