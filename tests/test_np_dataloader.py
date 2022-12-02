import functools
import unittest
import logging

import numpy as np

from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.data import NPDataLoader


class TestNPDataLoader(unittest.TestCase):
    def setUp(self):
        self.len = 7
        self.X = np.arange(7)[:, np.newaxis]
        logging.disable('CRITICAL')

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def assertIterableEqual(self, it1, it2):
        self.assertListEqual([x for x in it1], [x for x in it2])

    def assertNPArrayEqual(self, arr1, arr2):
        self.assertIterableEqual(arr1.flatten(), arr2.flatten())

    def iterate_and_assert(self, batch_size, n_epochs, drop_last=False):
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size,
                                  drop_last=drop_last)

        num_batches_per_epoch = self.len//batch_size
        if not drop_last and self.len % batch_size > 0:
            num_batches_per_epoch += 1

        self.assertEqual(num_batches_per_epoch, len(dataloader))

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

    def test_npdataloader_00_creation(self):
        _ = NPDataLoader(dataset=self.X,
                         target=self.X)  # Base case that should not raise errors

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset='wrong-type',
                             target=self.X)

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset=self.X,
                             target='wrong-type')

        # test that inconsistent lengths raise ValueError
        with self.assertRaises(FedbiomedValueError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X[:2, :])

        # test that 1-d targets are handled correctly
        loader = NPDataLoader(dataset=np.squeeze(self.X),
                              target=np.squeeze(self.X))
        self.assertIterableEqual(loader.dataset.shape, loader.target.shape)

        # test that wrong dataset shape raises ValueError
        with self.assertRaises(FedbiomedValueError):
            _ = NPDataLoader(dataset=self.X[:, np.newaxis],
                             target=self.X)

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             batch_size='wrong-type')

        with self.assertRaises(FedbiomedValueError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             batch_size=-1)

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             drop_last='wrong-type')

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             shuffle='wrong-type')

        with self.assertRaises(FedbiomedTypeError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             random_seed='wrong-type')

        # ensure that an unknown argument raises TypeError (this is used in SKLearnDataManager.split)
        with self.assertRaises(TypeError):
            _ = NPDataLoader(dataset=self.X,
                             target=self.X,
                             unknown_argument='unknown')

    def test_npdataloader_01_iteration(self):
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

        # test iteration with targets
        batch_size = 2
        dataloader = NPDataLoader(dataset=self.X,
                                  target=3.*self.X + 1.,
                                  batch_size=batch_size)
        num_batches_per_epoch = self.len // batch_size + 1  # drop_last is False
        n_epochs = 2
        outcome = list()
        for epoch in range(n_epochs):
            for data, target in dataloader:
                outcome.append((data, target))

        self.assertEqual(len(outcome), num_batches_per_epoch*n_epochs)

        expected_data_sum = self.X.sum()
        data_sum = functools.reduce(lambda x, y: x + y[0].sum(), outcome, 0)
        self.assertEqual(data_sum, expected_data_sum * n_epochs)

        expected_target_sum = (3.*self.X + 1).sum()
        target_sum = functools.reduce(lambda x, y: x + y[1].sum(), outcome, 0)
        self.assertEqual(target_sum, expected_target_sum * n_epochs)

    def test_npdataloader_02_shuffle(self):
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=1,
                                  shuffle=True,
                                  random_seed=42)
        outcome = list()
        for data, target in dataloader:
            outcome.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, self.X)]))

        # Assert that iterating a second time yields another shuffling
        second_epoch = list()
        for data, target in dataloader:
            second_epoch.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, second_epoch)]))


    def test_npdataloader_03_target(self):
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

    def test_npdataloader_04_empty(self):
        """Test that NPDataLoader correctly handles empty arrays.

        The behaviour for empty arrays is:
        - NPDataLoader should not fail
        - The iterator should immediately raise StopIteration
        - len should be 0
        """
        dataloader = NPDataLoader(dataset=np.array([]),
                                  target=np.array([]),
                                  batch_size=2,
                                  shuffle=True,
                                  drop_last=True)

        for epoch in range(2):
            count = 0
            for i, (_, _) in enumerate(dataloader):
                count += 1
            self.assertEqual(count, 0)

        self.assertEqual(epoch, 1)
        self.assertEqual(len(dataloader), 0)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
