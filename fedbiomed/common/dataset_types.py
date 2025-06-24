# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes for dataset's data types and structures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fedbiomed.common.constants import _BaseEnum


class DataReturnFormat(_BaseEnum):
    """Possible return formats of data samples by dataset and reader classes"""

    # for a Dataset: DEFAULT is generic Dataset data format
    # for a Reader: DEFAULT is native Reader data format
    DEFAULT = 0
    TORCH = 1
    SKLEARN = 2


drf_default = DataReturnFormat(DataReturnFormat.DEFAULT)


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


class DataType(_BaseEnum):
    """Possible data modality types"""

    IMAGE = 1
    TABULAR = 2


@dataclass
class DatasetDataItemModality:
    modality_name: str
    type: DataType
    # Q: are these exacly the types we want to use for generic format
    data: Union[np.ndarray, pd.DataFrame]


# Base type for `Dataset.__getitem__()` returning data in generic format
# a sample is `(DatasetDataItem, DatasetDataItem)` for `(data, target)`
#
# - DatasetDataItemModality when data is an array like in a generic format
# - Any when data is an array like in framework specific format (using `to_xxx`)
DatasetDataItem = Optional[Dict[str, Union[Any, DatasetDataItemModality]]]


# shape for a sample of *one* modality as a list of dimensions of array
ModalityShape = List[int]

# shape for a sample of a `Reader`'s returned item as
# `(number_of_samples, { 'modality_name' => dimensions_of_array })`
# This is needed to support the multiple cases, eg
# - a CSV may be interpreted as one sample (with multiple lines)
#   or multiple samples (one per line) => not done by the reader
# - data may be multi-modal (eg: BIDS)
ReaderShape = Dict[str, ModalityShape]

DatasetItemShape = Optional[Dict[str, Tuple[DataType, ModalityShape]]]
# shape for a full sample as `(number of samples, data shape, target shape)`
DatasetShape = Tuple[int, DatasetItemShape, DatasetItemShape]


@dataclass
class DatasetDataModality:
    """Structure and metadata of a dataset's modality"""

    modality_name: str
    type: DataType
    shape: ModalityShape
    # add more fields in the future for federated analytics function overload, metadata, etc.


# Describe the structure and metadata of the full dataset
# does not contain the data or the `Reader`s


# Nota: `len` is number of samples of dataset. May be different than number
# of sample in a modality if we have incomplete samples, and remove with
# dataset-specific rules
#
# Nota: value depend on DLPs (filter modality, samples)
@dataclass
class DatasetData(ABC):
    data: dict[str, DatasetDataModality]
    target: dict[str, DatasetDataModality]
    len: int

    @abstractmethod
    def _dummy(self):
        """Dummy abstract method to prevent instantiation of base class"""


# Nota: keep distinction between native and structured for future extensions


class DatasetDataNative(DatasetData):
    def _dummy(self):
        """Dummy method to enable instantiation of class"""
        pass


class DatasetDataStructured(DatasetData):
    def _dummy(self):
        """Dummy method to enable instantiation of class"""
        pass
