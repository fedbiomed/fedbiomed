# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fedbiomed.common.constants import ErrorNumbers, _BaseEnum
from fedbiomed.common.exceptions import FedbiomedError


# ENUMS =======================================================================
class DataReturnFormat(_BaseEnum):
    """Possible return formats of data samples by dataset and reader classes"""

    DEFAULT = 0
    TORCH = 1
    SKLEARN = 2


class ModalityType(_BaseEnum):
    IMAGE = 1
    TABULAR = 2


# DATACLASSES =================================================================
@dataclass
class ItemModality:
    name: str
    type: ModalityType
    data: Union[np.ndarray, pd.DataFrame]


@dataclass
class MetadataModality:
    name: str
    type: ModalityType
    shape: Tuple[int, ...]


# TYPES =======================================================================
SampleModality = tuple[Dict[str, ItemModality], Optional[Dict[str, ItemModality]]]
Transform = Optional[Union[Callable, Dict[str, Callable]]]


# CLASSES =====================================================================
class Dataset(ABC):
    _to_format: DataReturnFormat = DataReturnFormat.DEFAULT

    # PROPERTIES ==============================================================
    # root ====================================================================
    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, path_input: Union[str, Path]):
        """Root setter

        Raises:
            FedbiomedError:
            - if root type is not str or pathlib.Path
            - if root does not exist
        """
        if not isinstance(path_input, (str, Path)):
            raise FedbiomedError(
                ErrorNumbers.FB632.value
                + ": Expected a string or Path, got "
                + type(path_input).__name__
            )
        path = Path(path_input).expanduser().resolve()
        if not path.exists():
            raise FedbiomedError(
                ErrorNumbers.FB632.value + ": Path does not exist, " + str(path)
            )
        self._root = path

    # transforms ==============================================================
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

    # ABSTRACT FUNCTIONS ======================================================
    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Retrieve a data sample"""
        pass

    @abstractmethod
    def validate(self):
        pass

    # FUNCTIONS ===============================================================
    def to_generic(self) -> None:
        self._to_format = DataReturnFormat.DEFAULT

    def _get_dataset_data_meta(self):
        return tuple(
            {
                item.name: MetadataModality(
                    name=item.name,
                    type=item.type,
                    shape=item.data.shape,
                )
                for item in self._get_generic_format_item(0)
            }
        )
