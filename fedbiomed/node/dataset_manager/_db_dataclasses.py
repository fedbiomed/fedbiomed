# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataclasses for database entities
"""

import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Union, get_origin

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class TableEntry:
    """Base class for database entries with common conversion methods"""

    def to_dict(self) -> dict:
        """Convert entry to dictionary - removes None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def validate_required_fields(cls, data: dict) -> None:
        """Validate that all required fields are present"""
        # Auto-extract required fields from dataclass definition
        required_fields = [
            field.name
            for field in fields(cls)
            if (
                field.default == field.default_factory
                and get_origin(field.type) not in (Union, type(Union[str, None]))
            )
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Validation failed for {cls.__name__}: "
                f"Missing required fields: {missing_fields}"
            )

    @classmethod
    def from_dict(cls, data: dict):
        """Generic from_dict implementation for all dataclasses"""
        cls.validate_required_fields(data)

        # Get all field names and their default values
        field_mapping = {field.name: field for field in fields(cls)}

        # Build kwargs for dataclass constructor
        kwargs = {}
        for field_name, field_obj in field_mapping.items():
            if field_name in data:
                kwargs[field_name] = data[field_name]
            elif field_obj.default != field_obj.default_factory:
                # Field has a default value, use .get() to get default
                kwargs[field_name] = data.get(field_name, field_obj.default)
            else:
                # Optional field without default
                kwargs[field_name] = data.get(field_name)

        return cls(**kwargs)


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
