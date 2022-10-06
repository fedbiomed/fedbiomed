import functools
import unittest

import numpy as np

from fedbiomed.common.data import NPDataLoader


class TestNPDataLoader(unittest.TestCase):
    def setUp(self):
        self.len = 7
        self.X = np.arange(7)[:, np.newaxis]

    def assertIterableEqual(self, it1, it2):
        self.assertListEqual([x for x in it1], [x for x in it2])

    def assertNPArrayEqual(self, arr1, arr2):
        self.assertIterableEqual(arr1.flatten(), arr2.flatten())

    def iterate_and_assert(self, batch_size, n_epochs, drop_last=False):
        dataloader = NPDataLoader(dataset=self.X, batch_size=batch_size, drop_last=drop_last)

        num_batches_per_epoch = self.len//batch_size
        if not drop_last and self.len % batch_size > 0:
            num_batches_per_epoch += 1

        self.assertEqual(num_batches_per_epoch, dataloader.get_num_batches())

        outcome = list()
        for epoch in range(n_epochs):
            for data, target in dataloader:
                outcome.append(data)

        self.assertEqual(len(outcome), num_batches_per_epoch*n_epochs)

        remainder = self.len % batch_size
        if remainder > 0 and drop_last:
            expected_sum = self.X[:-remainder].sum()
        else:
            expected_sum = self.X.sum()
        out_sum = functools.reduce(lambda x, y: x + y.sum(), outcome, 0)
        self.assertEqual(out_sum, expected_sum*n_epochs)

        if remainder == 0:
            self.assertNPArrayEqual(outcome[-1], self.X[-batch_size:])
        else:
            if not drop_last:
                self.assertNPArrayEqual(outcome[-1], self.X[-remainder:])
            else:
                self.assertNPArrayEqual(outcome[-1], self.X[-batch_size-remainder:-remainder])

    def test_npdataloader_NN_iteration(self):
        # scenario: batch_size=1 and 1 epoch
        self.iterate_and_assert(1, 1, drop_last=False)
        # scenario: batch_size=full dataset and 1 epoch
        self.iterate_and_assert(self.len, 1, drop_last=False)
        # scenario: batch_size=3 and 1 epoch
        self.iterate_and_assert(3, 1, drop_last=False)

        # same scenarios, but 3 epochs
        self.iterate_and_assert(1, 3, drop_last=False)
        self.iterate_and_assert(self.len, 3, drop_last=False)
        self.iterate_and_assert(3, 3, drop_last=False)

        # same as initial scenarios, but drop_last=True
        self.iterate_and_assert(1, 1, drop_last=True)
        self.iterate_and_assert(self.len, 1, drop_last=True)
        self.iterate_and_assert(3, 1, drop_last=True)

        # same as initial scenarios, but 3 epochs and drop_last=True
        self.iterate_and_assert(1, 3, drop_last=True)
        self.iterate_and_assert(self.len, 3, drop_last=True)
        self.iterate_and_assert(3, 3, drop_last=True)

    def test_npdataloader_NN_shuffle(self):
        dataloader = NPDataLoader(dataset=self.X, batch_size=1, shuffle=True, random_seed=42)
        outcome = list()
        for data, target in dataloader:
            outcome.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, self.X)]))

        # Assert that iterating a second time yields another shuffling
        second_epoch = list()
        for data, target in dataloader:
            second_epoch.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, second_epoch)]))


    def test_npdataloader_NN_target(self):
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=1,
                                  random_seed=42)
        for epoch in range(2):
            for data, target in dataloader:
                self.assertEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  random_seed=42)
        for epoch in range(2):
            for data, target in dataloader:
                self.assertNPArrayEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  drop_last=True,
                                  random_seed=42)
        for epoch in range(2):
            for data, target in dataloader:
                self.assertNPArrayEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  drop_last=True,
                                  shuffle=True,
                                  random_seed=42)
        for epoch in range(2):
            for data, target in dataloader:
                self.assertNPArrayEqual(data, target)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
