"""DataLoaders for Fed-BioMed.

DataLoader classes are responsible for:
- calling the dataset's `__getitem__` method when needed
- collating samples in a batch
- shuffling the data at every epoch
- in general, managing the actions related to iterating over a certain dataset

DataLoaders in Fed-BioMed view the dataset as an infinite stream of samples.
Please refer to the [DataLoader user guide](/user-guide/researcher/training-data/#data-loaders)
for an introduction.

Fed-BioMed support iterating through a dataset in a standard way, very similar to Torch DataLoader API. However,
Fed-BioMed DataLoaders do not raise `StopIteration` after one epoch. Instead, they will continue iterating indefinitely.

!!! warning "DataLoaders do not automatically stop at the end of an epoch"
    Instead, the developer is required to keep track of the number of iterations that have been performed, and implement
    their own strategy for exiting the loop.

This was done because we want to "move away" from the notion of epochs, which in a federated setting does not make
much sense since each node may have a different number of samples in their dataset. In order to achieve this, we
need DataLoaders to be able to cycle through the data as many times as needed to achieve the total number of updates
requested by the researcher.
"""
from .np_dataloader import NPDataLoader

__all__ = [
    'NPDataLoader'
]
