# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for datasets
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
from pathlib import Path

from fedbiomed.common.dataset_types import DataReturnFormat, Transform, DatasetDataItem


class Dataset(ABC):

    _framework_transform : Transform = None
    _framework_target_transform : Transform = None

    # Implementation of generic transform (or not) is specific to each dataset
    _generic_transform : Transform = None
    _generic_target_transform : Transform = None

    # Possible implementation
    #
    # Format for returning data sample, as some processing will change
    _to_format : DataReturnFormat = DataReturnFormat.GENERIC

    def __init__(
            self,
            framework_transform : Transform = None,
            framework_target_transform : Transform = None,
            # Optional, per-dataset: implement (or not) generic transform (use same argument name)
            # generic_transform : Transform = None,
            # generic_target_transform : Transform = None,
            # Optional, per dataset: implement native transforms (argument name may vary)
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        # TODO: check type
        self._framework_transform = framework_transform
        self._framework_target_transform = framework_target_transform


    def framework_transform(self) -> Transform:
        """Getter for framework transform"""
        return self._framework_transform

    def framework_target_transform(self) -> Transform:
        """Getter for framework target transform"""
        return self._framework_target_transform


    # Caveat: give explicit user error message when raising exception
    # Also need to wrap with try/except when calling `Reader` Transform (native Transform)
    # to give an explicit message (from the Dataset class)

    def _apply_generic_transform(
            self,
            sample: Tuple[DatasetDataItem, DatasetDataItem],
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Apply generic (target) Transform to a data sample in generic format

        Raises exception if cannot apply Transform to data
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: use Controller._get_nontransformed_item
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""


    # Optional methods which can be implemented (or not) by some datasets
    # Possible alternate implementation: class to be inherited by datasets that implement it
    # (multiple inheritance).

    # def to_torch(self) -> None:
    #     """Request dataset to return samples for a torch training plan
    # 
    # Ignore + issue warning if generic transform needs to be applied
    # """ 
    # 
    # def to_sklearn(self) -> None:
    #     """Request dataset to return samples for a sklearn training plan
    # 
    # Ignore + issue warning if generic transform needs to be applied
    # """

    # Nota: we could also implement a `to_native()` method in every dataset/reader,
    # but do we have a use case for that ?

    # Still needed or replaced by implementation in DLP ? cf current MedicalFolderDataset
    #
    # def set_dataset_parameters(self, parameters: dict):


    # Additional methods for exploring data (folders, modalities, subjects),
    # depending on Dataset and Reader


class StructuredDataset(Dataset):
    def __init__(
            self,
            root: Path,
            framework_transform : Transform = None,
            framework_target_transform : Transform = None,
            # Optional, per-dataset: implement (or not) generic transform (use same argument name)
            # generic_transform : Transform = None,
            # generic_target_transform : Transform = None,
            # Optional, per dataset: implement native transforms (argument name may vary)
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        super().__init__(framework_transform, *args, **kwargs)
