# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for Pytorch training plan
"""

from typing import Optional, Tuple

from fedbiomed.common.dataloader import PytorchDataLoader
from fedbiomed.common.dataset import Dataset

from ._framework_data_manager import FrameworkDataManager


class NewTorchDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for Pytorch training plans"""

    _dataset: Dataset

    def __init__(self, dataset: Dataset, **kwargs: dict):  # noqa : B027 # not yet implemented
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader
        """

    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[PytorchDataLoader, Optional[PytorchDataLoader]]:
        """Split dataset and return data loaders"""

    # Other methods from current TorchDataManager
    # - REMOVE to_sklearn()
    # - maybe can factor some methods to FrameworkDataManager ?
    # - make some methods private ?
