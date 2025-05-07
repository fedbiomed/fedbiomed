from abc import abstractmethod
import csv
from functools import cache
import numpy as np
import pandas as pd
from monai.transforms import LoadImage, ToTensor, Compose, ToNumpy
from monai.data import ITKReader
from typing import Dict
import torch

class GenericReader:
    # usually implementation is defined in `Node.DatasetManager`
    def __init__(self):
        self._transform_framework = lambda x: x
    @abstractmethod
    def read(self, path):
        """"""



class ImageReader(GenericReader):
    def __init__(self):
            
        self._reader = Compose( [
            LoadImage(ITKReader(), image_only=True)]
        )
        

    def read(self, path: str, **kwargs):
        return self._transform_framework(self._reader(path, **kwargs))

    def to_torch(self):
        # see if we are keeping things this way
        self._transform_framework = ToTensor()

    def to_sklearn(self):
        # FIXME: should we convert images into vectors for sklearn?
        # and do it here?
        self._transform_framework = ToNumpy()

class CSVReader(GenericReader):
    def __init__(self):
        
        self._reader = pd.read_csv
        self._index_col = None
        self._dataframe = None

    @cache
    def _read(self,path, index_col, **kwargs):
        sniffer = csv.Sniffer()
        with open(path, 'r') as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            file.seek(0)
            header = 0 if sniffer.has_header(file.read()) else None
        self._dataframe = self._reader(path, index_col=index_col, sep=delimiter, header=header, engine='python')
    
    def read(self, path, index_col=None, **kwargs):
        if self._dataframe is None:
            self._read(path, index_col, **kwargs)
        return self._dataframe
        return self._reader(path, **kwargs)
    
    def read_single_entry(self, path, entry: str, **kwargs) -> Dict:
        if self._dataframe is None:
            self._read(path, **kwargs)
        demographics = self._dataframe.loc[~self._dataframe.index.duplicated(keep="first")]

        return demographics.loc[entry].to_dict()
    
    def get_index(self):
        
        return self._dataframe.index
    
    def convert(self, data: Dict):
        # extras step for converting data using transforms
        """"""
        return self._transform_framework(data)

    def to_torch(self,):

        self._transform_framework = lambda x: torch.as_tensor(x)

    def to_sklearn(self):

        def method(x):
            if x:
                return np.array(x)
        self._transform_framework = method
        