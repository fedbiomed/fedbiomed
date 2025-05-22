# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Torch tabulated data manager
"""

from typing import Callable, Union, Tuple

import numpy as np
import pandas as pd

from torch import from_numpy, Tensor
from torch.utils.data import Dataset

from fedbiomed.common.data.readers import CSVReader
from fedbiomed.common.exceptions import FedbiomedDatasetError
from fedbiomed.common.constants import ErrorNumbers, DatasetTypes


class CSVDataset:
    def __init__(self, root:str = None, inputs=None, targets=None):
        self._csv_reader = CSVReader()
        self._root = root
        self._inputs = inputs
        self._targets = targets
        self._length = len(self._csv_reader.read(self._root)) if self._root else len(self._inputs)
        #self._transform_framework: Callable = lambda x:x
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, item: int):

        if self._inputs is not None:
            if self._targets is not None:
                return self._csv_reader._transform_framework(self._inputs), self._transform_framework(self._targets)
            else:
                return self._csv_reader._transform_framework(self._inputs)
        data = self._csv_reader.read(self._root)
        return self._csv_reader._transform_framework(data)


    def to_torch(self):
        self._csv_reader.to_torch()

    def to_sklearn(self):
        self._csv_reader.to_sklearn()
