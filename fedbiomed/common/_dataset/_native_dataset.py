# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for native dataset
"""

from typing import Any, Optional, Tuple

from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DatasetDataItem, Transform


class NativeDataset(Dataset):
    def __init__(
        self,
        data: Any,
        # Target is optional
        target: Optional[Any] = None,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        # Optional, per-dataset: implement (or not) generic transform (use same argument name)
        # generic_transform : Transform = None,
        # generic_target_transform : Transform = None,
        # Optional, per dataset: implement reader transforms (argument name may vary)
        *args,
        **kwargs,
    ) -> None:
        """Class constructor"""
        super().__init__(framework_transform, *args, **kwargs)

    # Implement abstract methods

    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: use Controller._get_nontransformed_item
    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""

    # Additional methods for exploring data (folders, modalities, subjects) ?
