# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for MNIST
"""

from typing import Tuple

from fedbiomed.common.dataset_types import Transform, DatasetDataItem

from fedbiomed.common.dataset_controller import MnistController
from ._dataset import StructuredDataset


class MnistDataset(StructuredDataset, MnistController):
    def __init__(
        self,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        # Do we want generic transforms for MNIST ?
        # generic_transform : Transform = None,
        # generic_target_transform : Transform = None,
    ) -> None:
        """Class constructor"""

    # Implement abstract methods

    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: use Controller._get_nontransformed_item
    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""

    # Support returning samples in format for torch training plan
    #
    # Nothing much to do in that case, just call `Reader` ?
    def to_torch(self) -> None:
        """Request dataset to return samples for a torch training plan

        Ignore + issue warning if generic transform needs to be applied
        """

    # Additional methods for exploring data (folders, modalities, subjects),
    # depending on Dataset and Reader
