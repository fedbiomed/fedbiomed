"""
Classes that simplify imports from fedbiomed.common.data
"""


from ._data_manager import DataManager
from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import SkLearnDataManager
from ._tabular_dataset import TabularDataset
from ._medical_datasets import NIFTIFolderDataset, MedicalFolderDataset, MedicalFolderBase, MedicalFolderController, \
    MedicalFolderLoadingBlockTypes
from ._flamby_dataset import FlambyDatasetMetadataBlock, FlambyLoadingBlockTypes, \
    FlambyDataset, discover_flamby_datasets
from ._data_loading_plan import (DataLoadingBlock,
                                 MapperBlock,
                                 DataLoadingPlan,
                                 DataLoadingPlanMixin,
                                 SerializationValidation  # keep it for documentation
                                 )

__all__ = [
    "MedicalFolderBase",
    "MedicalFolderController",
    "MedicalFolderDataset",
    "MedicalFolderLoadingBlockTypes",
    "DataManager",
    "TorchDataManager",
    "SkLearnDataManager",
    "TabularDataset",
    "NIFTIFolderDataset",
    "DataLoadingBlock",
    "MapperBlock",
    "DataLoadingPlan",
    "DataLoadingPlanMixin",
    "SerializationValidation",
    "FlambyDatasetMetadataBlock",
    "FlambyLoadingBlockTypes",
    "FlambyDataset",
    "discover_flamby_datasets"
]
