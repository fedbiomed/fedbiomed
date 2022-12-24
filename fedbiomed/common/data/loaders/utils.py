"""Utility functions for DataLoader classes."""
from typing import Union, Any
from torch.utils.data import DataLoader
from fedbiomed.common.data.loaders import NPDataLoader


def _generate_roughly_one_epoch(dataloader: Union[NPDataLoader, DataLoader]) -> Any:
    """Generator for roughly one epoch of data.

    Precisely, it is a generator that will stop iterating after len(dataloader) iterations.
    Note that this only generates exactly one epoch in the case where the batch size divides the dataset length.
    In all other cases, this will either drop the last batch (if drop_last=True), or yield a final batch populated with
    additional samples resampled from the dataset (if drop_last=False).

    Args:
        dataloader: an instance of a Fed-BioMed DataLoader (either NPDataLoader or torch.utils.data.DataLoader)
    """
    for i, x in enumerate(dataloader, start=1):
        if i > len(dataloader):
            break
        yield x
