# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for readers
"""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

from fedbiomed.common.dataset_types import DataReturnFormat, Transform, ReaderItemShape


class Reader(ABC):
    # Implementation of native transform (or not) is specific to each reader
    _native_transform: Transform = None
    _native_target_transform: Transform = None

    # Possible implementation
    #
    # Track format for returning reader data sample, as some processing will change
    #
    # Nota: `_to_format` also kept in `Dataset`. To avoid duplicating state variable,
    # we may prefer to implement passing this value as an argument for each `__getitem__` call
    _to_format: DataReturnFormat = DataReturnFormat.GENERIC

    def __init__(  # noqa : B027  # method empty for now, not yet implemented
        self,
        root: Path,
        # Optional, per-reader: implement (or not) native transform (use same argument name)
        # native_transform : Transform = None,
        # native_target_transform : Transform = None,
        *args,
        **kwargs,
    ) -> None:
        """Class constructor"""

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Retrieve a data sample"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    @abstractmethod
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

    # Nota: does not include filtering of DLP, which is unknown to Reader
    @abstractmethod
    def shape(self) -> ReaderItemShape:
        """Returns shape of a data modality served by a reader"""

    # Optional methods which can be implemented (or not) by some readers
    # Code is specific to each reader

    # This is needed by MedicalFolderDataset (and probably most multimodal datasets)
    # to coordinate *which* subject they retrieve next. "Next" with `__getitem__` for
    # a `Reader` for one modality does not address same subject as "next" for another
    # `Reader` of another modality. Thus only dataset can ensure coherence
    # of "next" sample retrieval, using a tag.
    #
    # def getitem_by_tag(self, tag: str) -> Any:
    #    """Retrieve a data sample identified by an arbitrary string tag"""

    # def to_torch(self) -> None:
    #     """Request reader to return samples for a torch training plan
    # """
    #
    # def to_sklearn(self) -> None:
    #     """Request reader to return samples for a sklearn training plan
    # """

    # Nota: we could also implement a `to_native()` method in every dataset/reader,
    # but do we have a use case for that ?

    # Additional methods for exploring data, depending on Reader
