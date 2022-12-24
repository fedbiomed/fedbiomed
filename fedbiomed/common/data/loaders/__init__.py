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

The recommended way to achieve this is the following (note the use of the start option in enumerate, and the quick
exit based on the iteration counter value):
```python
for i, (data, target) in enumerate(dataloader, start=1):
    # quick exit when total number of iterations is reached
    if i > num_updates:
        break
    # continue with rest of loop
    # e.g. perform training
```

This was done because we want to "move away" from the notion of epochs, which in a federated setting does not make
much sense since each node may have a different number of samples in their dataset. In order to achieve this, we
need DataLoaders to be able to cycle through the data as many times as needed to achieve the total number of updates
requested by the researcher.

Calling `__next__` on a DataLoader returns one batch of data. It is important to note that this batch will *always
contain `batch_size` elements*, regardless of whether some samples in the batch already belong to a new epoch.
Consider the following example:

- `dataset = [0, 1, 2, 3, 4]`
- `batch_size = 3`

then the first iteration yields `[0, 1, 2]`, and the second iteration yields `[3, 4, 0]`.
If shuffling was activated, then the second batch would be assembled as follows:

1. take the last two elements that haven't been used yet for this epoch
2. shuffle the data
3. complete the batch with the final missing element

At the moment Fed-BioMed provides a custom implementation for numpy-based datasets
(the [`NPDataLoader`][fedbiomed.common.data.loaders.NPDataLoader]) while torch-based datasets should rely on the
[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class.
"""
from .np_dataloader import NPDataLoader
from .torch_dataloader import CyclingRandomSampler, CyclingSequentialSampler
from .utils import _generate_roughly_one_epoch

__all__ = [
    'NPDataLoader',
    'CyclingRandomSampler',
    'CyclingSequentialSampler',
    '_generate_roughly_one_epoch'
]
