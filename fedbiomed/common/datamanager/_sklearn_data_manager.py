# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for scikit-learn training plan
"""

from typing import Optional, Tuple

from fedbiomed.common.dataloader import SkLearnDataLoader
from fedbiomed.common.dataset import Dataset

from ._framework_data_manager import FrameworkDataManager


class SkLearnDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for scikit-learn training plans"""

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
    ) -> Tuple[SkLearnDataLoader, Optional[SkLearnDataLoader]]:
        """Split dataset and return data loaders"""

    # Other methods from current TorchDataManager
    # - maybe can factor some methods to FrameworkDataManager ?
    # - make some methods private ?
