# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract class for datasets
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Dict, Any
from pathlib import Path


class Dataset(ABC):

    _framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None

    # Implementation of generic transform (or not) is specific to each dataset
    _generic_transform : Optional[Union[Callable, Dict[str, Callable]]] = None

    def __init__(
            self,
            framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        if framework_transform:
            # TODO: check type
            self._framework_transform = framework_transform

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples"""

    # TODO: define DatasetDataItem structure

    @abstractmethod
    def __getitem__(self, key: int) -> DatasetDataItem:
        """Retrieve a data sample"""

    def framework_transform(self) -> Optional[Union[Callable, Dict[str, Callable]]]:
        """Getter for framework transform"""
        return self._framework_transform

    # Optional methods which can be implemented (or not) by every dataset

    # def to_torch(self) -> None:
    #     """Request dataset to return samples for a torch training plan""" 
    # 
    # def to_sklearn(self) -> None:
    #     """Request dataset to return samples for a sklearn training plan"""

    # Nota: we could also implement a `to_native()` method in every dataset/reader,
    # but do we have a use case for that ?


class NativeDataset(Dataset):
    def __init__(
            self,
            data: Any,
            framework_transform : Optional[Union[Callable, Dict[str, Callable]]] = None,
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
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""
        super().__init__(framework_transform, *args, **kwargs)
