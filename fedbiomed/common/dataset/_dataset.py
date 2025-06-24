# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for datasets
"""

from abc import ABC, abstractmethod
from typing import Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem, Transform
from fedbiomed.common.exceptions import FedbiomedError


class Dataset(ABC):
    # Implementation of generic transform (or not) is specific to each dataset
    _generic_transform: Transform = None
    _generic_target_transform: Transform = None

    # Possible implementation
    #
    # Format for returning data sample, as some processing will change
    _to_format: DataReturnFormat = DataReturnFormat.DEFAULT

    def __init__(
        self,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        # See subclass: either `root` or `dataset` + `target`
        # Optional, per-dataset: implement (or not) generic transform (use same argument name)
        # generic_transform : Transform = None,
        # generic_target_transform : Transform = None,
        # Optional, per dataset: implement native transforms (argument name may vary)
        **kwargs,
    ) -> None:
        self.framework_transform = framework_transform
        self.framework_target_transform = framework_target_transform
        super().__init__(**kwargs)

    @property
    def framework_transform(self):
        return self._framework_transform

    @framework_transform.setter
    def framework_transform(self, transform_input: Transform):
        self._validate_transform_input(transform_input)
        self._framework_transform = transform_input

    @property
    def framework_target_transform(self):
        return self._framework_target_transform

    @framework_target_transform.setter
    def framework_target_transform(self, transform_input: Transform):
        self._validate_transform_input(transform_input)
        self._framework_target_transform = transform_input

    def _validate_transform_input(self, transform_input: Transform) -> None:
        """Raises FedbiomedError if transform_input is not a valid input"""
        if transform_input is None or callable(transform_input):
            return
        elif isinstance(transform_input, dict):
            if not all(
                isinstance(k, str) and callable(v) for k, v in transform_input.items()
            ):
                raise FedbiomedError(
                    ErrorNumbers.FB632.value
                    + ": Transform dict must map strings to callables"
                )
        else:
            raise FedbiomedError(
                ErrorNumbers.FB632.value + ": Unexpected Transform input"
            )

    # Caveat: give explicit user error message when raising exception
    # Also need to wrap with try/except when calling `Reader` Transform (native Transform)
    # to give an explicit message (from the Dataset class)

    def _apply_generic_transform(  # noqa : B027  # empty method for now
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

    # def to_torch(self) -> bool:
    #     """Request dataset to return samples for a torch training plan
    #
    # Return True if accepted by the dataset (no generic transform to apply)
    # Ignore + issue warning + return False if generic transform needs to be applied
    # """
    #
    # def to_sklearn(self) -> bool:
    #     """Request dataset to return samples for a sklearn training plan
    #
    # Return True if accepted by the dataset (no generic transform to apply)
    # Ignore + issue warning + return False if generic transform needs to be applied
    # """

    # Nota: we could also implement a `to_native()` method in every dataset,
    # but do we have a use case for that ?

    # Still needed or replaced by implementation in DLP ? cf current MedicalFolderDataset
    #
    # def set_dataset_parameters(self, parameters: dict):

    # Additional methods for exploring data (folders, modalities, subjects),
    # depending on Dataset and Reader


class StructuredDataset(Dataset):
    def __init__(
        self,
        # Optional, per-dataset: implement (or not) generic transform (use same argument name)
        # generic_transform : Transform = None,
        # generic_target_transform : Transform = None,
        # Optional, per dataset: implement reader transforms (argument name may vary)
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
