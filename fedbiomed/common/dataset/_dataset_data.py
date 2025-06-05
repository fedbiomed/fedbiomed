# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes for dataset's data types and structures
"""

from typing import Optional, Dict, Union, Any
from dataclasses import dataclass

from fedbiomed.common.constants import _BaseEnum


class DataType(_BaseEnum):
    """Possible data modality types"""
    IMAGE : int = 1
    TABULAR : int = 2


@dataclass
class DatasetDataItemModality:
    modality_name: str
    type: DataType
    data: Any


DatasetDataItem = Optional[Dict[str, Union[Any, DatasetDataItemModality]]] 
