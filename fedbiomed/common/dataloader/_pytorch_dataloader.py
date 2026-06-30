# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Class for data loader in PyTorch training plans
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import default_collate

from ._dataloader import DataLoader

# Base type for data returned by `PytorchDataLoader` iterator
# a sample is tuple `(PytorchDataItem, PytorchDataItem)` for `(data, target)`
PytorchDataLoaderItem = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
PytorchDataLoaderSample = Tuple[PytorchDataLoaderItem, PytorchDataLoaderItem]


def collate_optional_target(batch: List[PytorchDataLoaderSample]) -> Any:
    """Collate ``(data, target)`` pairs; return target ``None`` when every target
    is ``None`` (unsupervised), else apply the default collation unchanged."""
    if all(target is None for _, target in batch):
        return default_collate([data for data, _ in batch]), None
    return default_collate(batch)


# Same as TorchDataLoader, just ensure it is also typed as a DataLoader
class PytorchDataLoader(TorchDataLoader, DataLoader):
    """Data loader class for PyTorch training plan"""

    def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
        """Constructor defaulting to a collate that tolerates a ``None`` target."""
        kwargs.setdefault("collate_fn", collate_optional_target)
        super().__init__(dataset, *args, **kwargs)

    # No need to implement methods
    #
    #     def __init__(self, dataset: TorchDataset, *args, **kwargs) -> None:
    #         """Class constructor"""
    #         super().__init__(dataset, *args, **kwargs)
    #
    #     def __len__(self) -> int:
    #         """Returns the number of batches of the encapsulated dataset"""
    #
    #     def __iter__(self) -> _BaseDataLoaderIter:
    #         """Returns an iterator over batches of data"""

    # Torch implements as public variables, needed for its initialization
    # Don't overwrite with properties as it breaks the parent class
    #
    #     @property
    #     def dataset(self) -> TorchDataset:
    #         """Returns the encapsulated dataset"""
    #
    #     @property
    #     def batch_size(self) -> int:
    #         """Returns the batch size used by the data loader"""
    #
    #     @property
    #     def drop_last(self) -> bool:
    #         """Returns whether the data loader drops the last incomplete batch"""
