# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.datamanager
"""

from ._data_manager import DataManager
from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import SkLearnDataManager
from ._framework_data_manager import FrameworkDataManager
from ._new_data_manager import NewDataManager
from ._new_torch_data_manager import NewTorchDataManager
from ._new_sklearn_data_manager import NewSkLearnDataManager

__all__ = [
    "DataManager",
    "TorchDataManager",
    "SkLearnDataManager",
    "FrameworkDataManager",
    "NewDataManager",
    "NewTorchDataManager",
    "NewSkLearnDataManager",
]
