# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MNIST
"""
from pathlib import Path
from typing import Tuple, Union

from torchvision import datasets

from fedbiomed.common.dataset_types import DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError

from ._controller import Controller


class MnistController(Controller):
    def __init__(self, root: Union[str, Path], **kwargs) -> None:
        try:
            self.root = root

            # Download dataset if it does not exist
            dataset = datasets.MNIST(root=self.root, download=True)
            self._data = dataset.data
            self._targets = dataset.targets
            
            super().__init__(**kwargs)
        
        except Exception as e:
            raise FedbiomedError(e) from e
        
    def _get_nontransformed_item(
        self, index: int,
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        # data_item and target_item are torch tensors
        data_item = {"data": self._data[index]}
        target_item = {"target": self._targets[index]}
        return data_item, target_item

    def validate(self) -> None:
        # TODO
        pass
