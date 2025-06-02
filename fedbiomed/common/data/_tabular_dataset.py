# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Torch tabulated data manager
"""

from typing import Callable, List, Union, Tuple

import numpy as np
import pandas as pd

from torch import from_numpy, Tensor
from torch.utils.data import Dataset

from fedbiomed.common.data._generic_dataset import GenericDataset
from fedbiomed.common.data.readers import CSVReader
from fedbiomed.common.exceptions import FedbiomedDatasetError
from fedbiomed.common.constants import ErrorNumbers, DatasetTypes


class CSVDataset(GenericDataset):
    def __init__(self, root:str = None, inputs=None, targets=None):
        self._csv_reader = CSVReader()
        self._root = root
        self._inputs = inputs
        self._targets = targets
        self._length = len(self._csv_reader.read(self._root)) if self._root else len(self._inputs)
        #self._transform_framework: Callable = lambda x:x
        self._index = list(range(self._length))
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, item: int):

        if self._inputs is not None:
            if self._targets is not None:
                return {'data': self._csv_reader._transform_framework(self._inputs)[item],
                        'target': self._csv_reader._transform_framework(self._targets)[item]}
            else:
                return {'data': self._csv_reader._transform_framework(self._inputs)[item]}
        data = self._csv_reader.read(self._root)
        return {'data': self._csv_reader._transform_framework(data)[item]}, None

    def set_index(self, index: List[int]):
        self._index = index
    @classmethod
    def dataset_builder(cls, dataset:'CSVDataset', values, index: List[int], test_batch_size: int =None):
        _dataset = cls(dataset._root, 
                       dataset._inputs,
                       dataset._targets,)

        _dataset._csv_reader = dataset._csv_reader
        _dataset.set_index(index)
        return _dataset

    def to_torch(self):
        self._csv_reader.to_torch()

    def to_sklearn(self):
        self._csv_reader.to_sklearn()
