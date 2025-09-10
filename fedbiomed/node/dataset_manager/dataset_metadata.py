from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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

    def get_controller_arguments(self) -> Dict[str, Any]:
        """Get arguments to be passed to the dataset controller

        Returns:
            Dictionary of arguments
        """
        args = {
            "root": self.path,
        }
        if self.dataset_parameters:
            args.update(self.dataset_parameters)
        return args


@dataclass
class CsvMetadata(DatasetMetadata):
    dtypes: List[str]

    def __post_init__(self):
        self.dataset_parameters = {"dtypes": self.dtypes}


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

    def __post_init__(self):
        self.dataset_parameters = {
            "tabular_file": self.tabular_file,
            "index_col": self.index_col,
            "dlp_id": self.dlp_id,
        }
