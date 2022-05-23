"""
to simplify imports from fedbiomed.common.data
"""


from ._data_manager import DataManager
from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import SkLearnDataManager
from ._tabular_dataset import TabularDataset
from ._medical_datasets import NIFTIFolderDataset

__all__ = [
    "DataManager",
    "TorchDataManager",
    "SkLearnDataManager",
    "TabularDataset",
    "NIFTIFolderDataset"
]
