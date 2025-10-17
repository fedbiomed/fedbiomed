# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataclasses for database entities
"""

import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class TableEntry:
    """Base class for database entries with common conversion methods"""

    def to_dict(self) -> dict:
        """Convert entry to dictionary - removes None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        """Generic from_dict implementation for dataclasses"""
        try:
            return cls(**data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Invalid data input for {cls.__name__}: "
                f"{str(e)}"
            ) from e


@dataclass
class DatasetEntry(TableEntry):
    """Simplified dataset entry using single dataclass for all dataset types"""

    name: str
    data_type: str
    tags: List[str]
    description: str
    path: str
    shape: List[int] | Dict[str, List[int]]
    dtypes: Dict[str, str]

    dataset_id: Optional[str] = None
    dataset_parameters: Optional[Dict[str, Any]] = None
    dlp_id: Optional[str] = None

    def __post_init__(self):
        if self.dataset_id is None:
            self.dataset_id = f"dataset_{uuid.uuid4()}"


@dataclass
class DlpEntry(TableEntry):
    """Data Loading Plan entry"""

    dlp_id: str
    name: str
    target_dataset_type: str
    loading_plan_path: str
    desc: Optional[str] = None


@dataclass
class DlbEntry(TableEntry):
    """Data Loading Block entry"""

    dlb_id: str
    name: str
    dataset_id: str
    dlp_id: str
    loading_block_path: str
    desc: Optional[str] = None
