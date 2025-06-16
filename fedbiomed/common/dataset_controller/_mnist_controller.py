# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MNIST
"""

from pathlib import Path
from typing import Tuple

from fedbiomed.common.dataset_types import DatasetDataItem

from ._controller import Controller


class MnistController(Controller):
    def __init__(
        self,
        root: Path,
        # Any other parameter ?
    ) -> None:
        """Class constructor"""

    # Implement abstract methods

    def validate(self) -> None:
        pass

    def _get_nontransformed_item(
        self, index: int
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        pass

    # Additional methods for exploring data (folders, modalities, subjects),
    # depending on Reader
