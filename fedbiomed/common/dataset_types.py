# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes for dataset's data types and structures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from fedbiomed.common.constants import FedbiomedError, _BaseEnum


class DataReturnFormat(_BaseEnum):
    """Possible return formats of data samples by dataset and reader classes"""

    SKLEARN = np.ndarray
    TORCH = torch.Tensor


# Type for researcher-defined (in training plan) data transforms
#
# # OK: no framework transform
# framework_transform = None
#
# # OK: one callable used for all modalities
# framework_transform = ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
#
# # OK: one dict with an item defining the callable for each used modality
# framework_transform = {
#   'T1': ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
#   'T2': ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
# }
#
# # OK: one dict but missing item for some used modality
# # => will apply like None (identity) to this modality
# framework_transform = { 'T2': ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) }

Transform = Optional[Union[Callable, Dict[str, Callable]]]

# Base type for `Dataset.__getitem__()` returning data in framework format
# a sample is `(DatasetDataItem, DatasetDataItem)` for `(data, target)`
#
# `Any`` represents the data in the framework specific format (using `to_format`)
DatasetDataItem = Optional[Union[Any, Dict[str, Any]]]


# === Schema Types ===


# Type for statistics configuration (used in analytics)
# Can be a string ("mean") or a tuple ("histogram", {"bins": 10})
StatConfig = Union[str, Tuple[str, Dict[str, Any]]]


class DatasetElementType(_BaseEnum):
    """Classification of leaf data elements in a dataset sample (schema)."""

    IMAGE = "image"
    ROW = "row"
    UNKNOWN = "unknown"


@dataclass
class DatasetElementSpec(ABC):
    """Base class for dataset element specifications."""

    @property
    @abstractmethod
    def type(self) -> DatasetElementType:
        pass


@dataclass
class ImageSpec(DatasetElementSpec):
    """Specification for an image element."""

    @property
    def type(self) -> DatasetElementType:
        return DatasetElementType.IMAGE


@dataclass
class RowSpec(DatasetElementSpec):
    """Specification for a tabular row element."""

    columns: List[str]

    def __post_init__(self):
        if not self.columns:
            raise FedbiomedError("RowSpec must have at least one column.")

    @property
    def type(self) -> DatasetElementType:
        return DatasetElementType.ROW


@dataclass
class UnknownSpec(DatasetElementSpec):
    """Specification for an unknown element."""

    @property
    def type(self) -> DatasetElementType:
        return DatasetElementType.UNKNOWN
