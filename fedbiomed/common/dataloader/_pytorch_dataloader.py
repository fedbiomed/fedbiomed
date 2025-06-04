# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Class for data loader in PyTorch training plans
"""

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter 

from ._dataloader import DataLoader


class PytorchDataLoader(TorchDataLoader, DataLoader):
    # In that case
    # - either implement empy method (only inherit type DataLoader)
    # - or implement methods that call TorchDataLoader methods, cf below
    #   to be more explicit


    #def __init__(self, dataset: TorchDataset, *args, **kwargs) -> None:
    #    """Class constructor"""
#
    #@property
    #def dataset(self) -> TorchDataset:
    #    """Returns the encapsulated dataset"""
#
    #def __len__(self) -> int:
    #    """Returns the length of the encapsulated dataset"""
#
    #def __iter__(self) -> _BaseDataLoaderIter:
    #    """Returns an iterator over batches of data"""
