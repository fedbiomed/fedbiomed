# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for readers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    ReaderShape,
    Transform,
    drf_default,
)


class Reader(ABC):
    # Implementation of reader transform (or not) is specific to each reader
    _reader_transform: Transform = None
    _reader_target_transform: Transform = None

    # Possible implementation
    #
    # Track format for returning reader data sample, as some processing will change
    #
    # Nota: `_to_format` also kept in `Dataset`. To avoid duplicating state variable,
    # we may prefer to implement passing this value as an argument for each `__getitem__` call
    _to_format: DataReturnFormat = DataReturnFormat(DataReturnFormat.DEFAULT)

    def __init__(  # noqa : B027  # method empty for now, not yet implemented
        self,
        root: Path,
        to_format: DataReturnFormat = drf_default,
        # Optional, per-reader: implement (or not) reader transform (use same argument name)
        # reader_transform : Transform = None,
        # reader_target_transform : Transform = None,
        *args,
        **kwargs,
    ) -> None:
        """Class constructor"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    # Nota: some reader may support index or range when data has multiple entries (eg CSV)
    @abstractmethod
    def read(self) -> Any:
        """Retrieve data"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    @abstractmethod
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

    # Nota: does not include filtering of DLP, which is unknown to Reader
    @abstractmethod
    def shape(self) -> ReaderShape:
        """Returns shape of the data served by a reader

        Computed before applying transforms or conversion to other format
        """

    # Optional methods which can be implemented (or not) by some readers
    # Code is specific to each reader

    # def len(self) -> int:
    #    """Get number of samples"""

    # Additional methods for exploring data, depending on Reader
