import json
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union


@dataclass
class DatasetMetadata:
    name: str
    data_type: str
    tags: List[str]
    description: str
    shape: Union[List[int], Dict[str, List[int]]]
    path: str
    dataset_id: str
    dataset_parameters: Optional[Dict[str, Any]]


# Individual dataset classes (for clarity)
@dataclass
class CsvMetadata(DatasetMetadata):
    dtypes: List[str]
    pass


@dataclass
class MnistMetadata(DatasetMetadata):
    pass


@dataclass
class MednistMetadata(DatasetMetadata):
    pass


@dataclass
class ImagesMetadata(DatasetMetadata):
    pass


@dataclass
class MedicalFolderMetadata(DatasetMetadata):
    tabular_file: Optional[str]
    index_col: Optional[int]
    dlp_id: Optional[str]