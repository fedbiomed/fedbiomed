from itertools import cycle
from typing import Iterator
from torch.utils.data import SequentialSampler, RandomSampler, Sampler


class CyclingSequentialSampler(SequentialSampler):
    def __iter__(self) -> Iterator[int]:
        return cycle(super().__iter__())


class CyclingRandomSampler(Sampler[int]):
    class _Iter:
        def __init__(self, dataset, **kwargs):
            self._dataset = dataset
            self._kwargs = kwargs
            self._sampler = RandomSampler(self._dataset, **self._kwargs)
            self._iter = iter(self._sampler)
            self.num_yielded = 0

        def __next__(self):
            if self.num_yielded == len(self._sampler):
                self.reset()
            self.num_yielded += 1
            return next(self._iter)

        def reset(self):
            self.num_yielded = 0
            self._sampler = RandomSampler(self._dataset, **self._kwargs)
            self._iter = iter(self._sampler)

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self._kwargs = kwargs

    def __len__(self):
        return len(self._dataset)

    def __iter__(self) -> Iterator[int]:
        return CyclingRandomSampler._Iter(self._dataset, **self._kwargs)


