# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for datasets
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Dict, Any, Tuple
from pathlib import Path

from ._dataset_data import DatasetDataItem
# from fedbiomed.common.dataset_reader import Reader


class Dataset(ABC):

    _framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None
    _framework_target_transform : Optional[Union[Callable, Dict[str, Callable]]] = None

    # Implementation of generic transform (or not) is specific to each dataset
    _generic_transform : Optional[Union[Callable, Dict[str, Callable]]] = None
    _generic_target_transform : Optional[Union[Callable, Dict[str, Callable]]] = None

    # Possible implementation
    #
    # True if `to_torch()`` or `to_sklearn()` is active, as some processing will change
    # _to_framework : bool = False

    # Possible implementation
    # _readers : Dict[str, Reader]

    def __init__(
            self,
            framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            framework_target_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        # TODO: check type
        self._framework_transform = framework_transform
        self._framework_target_transform = framework_target_transform


    def framework_transform(self) -> Optional[Union[Callable, Dict[str, Callable]]]:
        """Getter for framework transform"""
        return self._framework_transform

    def framework_target_transform(self) -> Optional[Union[Callable, Dict[str, Callable]]]:
        """Getter for framework target transform"""
        return self._framework_target_transform


    # Caveat: give explicit user error message when raising exception
    # Also need to wrap with try/except when calling `Reader` transforms (native transforms)
    # to give an explicit message (from the Dataset class)

    def _apply_generic_transforms(
            self,
            sample: Tuple[DatasetDataItem, DatasetDataItem],
        ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Apply generic (target) transforms to a data sample in generic format

        Raises exception if cannot apply transforms to data
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples"""

    # TODO: define DatasetDataItem structure

    @abstractmethod
    def __getitem__(self, key: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""

    # Optional methods which can be implemented (or not) by every dataset
    # Possible implementation: class to be inherited by datasets that implement it
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


class NativeDataset(Dataset):
    def __init__(
            self,
            data: Any,
            framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            framework_target_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        super().__init__(framework_transform, *args, **kwargs)


class StructuredDataset(Dataset):
    def __init__(
            self,
            root: Path,
            framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            framework_target_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        super().__init__(framework_transform, *args, **kwargs)
