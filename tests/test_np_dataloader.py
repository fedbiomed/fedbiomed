import itertools
import unittest
import logging

import numpy as np

from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError
from fedbiomed.common.data.loaders import NPDataLoader, _generate_roughly_one_epoch


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

    def test_npdataloader_00_creation(self):
        """Test that constructor arguments are properly handled"""
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
        """Test that the iteration process works correctly for NPDataLoader. """
        batch_size = 2
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size)
        num_updates = 5
        outcome_data = list()
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            outcome_data.append(data)
            # Assert batch size is correctly respected
            self.assertEqual(data.shape[0], batch_size)

        # Assert that the dataset was linearly traversed and we correctly looped back to the beginning after
        # the first epoch was finished
        concatenated_outcome_data = np.concatenate(outcome_data)
        for x, d in zip(itertools.cycle(self.X), concatenated_outcome_data):
            self.assertEqual(x, d)

        # Assert we visited the correct number of samples
        expected_num_samples_visited = num_updates*batch_size
        self.assertEqual(concatenated_outcome_data.shape[0], expected_num_samples_visited)

    def test_npdataloader_02_iteration_drop_last(self):
        """Test that the iteration process works correctly for NPDataLoader, when drop_last=True.

        Scenario:
            batch_size = 2
            dataset size = 7

        The values above imply that the last sample should be dropped.
        """
        batch_size = 2
        num_updates = 5
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size,
                                  drop_last=True)
        outcome_data = list()
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            outcome_data.append(data)
            # Assert batch size is correctly respected
            self.assertEqual(data.shape[0], batch_size)

        # Assert that the dataset was linearly traversed and we correctly looped back to the beginning after
        # the first epoch was finished.
        concatenated_outcome_data = np.concatenate(outcome_data)
        for x, d in zip(itertools.cycle(self.X[:-1]), concatenated_outcome_data):
            self.assertEqual(x, d)

        # Even when drop_last=True, the number of samples visited is always the same, because there are never any
        # incomplete batches!
        expected_num_samples_visited = num_updates*batch_size
        self.assertEqual(concatenated_outcome_data.shape[0], expected_num_samples_visited)


    def test_npdataloader_03_shuffle(self):
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=1,
                                  shuffle=True,
                                  random_seed=42)
        outcome = list()
        num_updates = len(self.X)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            outcome.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, self.X)]))

        # Assert that iterating a second time produces another shuffling
        second_epoch = list()
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            second_epoch.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome, second_epoch)]))

        # Assert that iterating for two epochs produces shuffling in-between
        outcome = list()
        num_updates = 2*len(self.X)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            outcome.append(data)
        self.assertTrue(any([x != y for x, y in zip(outcome[:len(self.X)],
                                                    outcome[len(self.X):])]))

    def test_npdataloader_04_target(self):
        num_updates = 2*len(self.X)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=1,
                                  random_seed=42)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            self.assertEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  random_seed=42)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            self.assertNPArrayEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  drop_last=True,
                                  random_seed=42)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            self.assertNPArrayEqual(data, target)

        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=3,
                                  drop_last=True,
                                  shuffle=True,
                                  random_seed=42)
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            self.assertNPArrayEqual(data, target)

    def test_npdataloader_05_empty(self):
        """Test that NPDataLoader correctly handles empty arrays.

        The behaviour for empty arrays is:
        - NPDataLoader should not fail
        - The iterator should cycle indefinitely
        - len should be 0
        """
        dataloader = NPDataLoader(dataset=np.array([]),
                                  target=np.array([]),
                                  batch_size=2,
                                  shuffle=True,
                                  drop_last=True)

        num_updates = 5
        count = 0
        for i, (data, target) in enumerate(dataloader, start=1):
            if i > num_updates:
                break
            count += 1

        self.assertEqual(count, num_updates)
        self.assertEqual(len(dataloader), 0)

    def test_npdataloader_06_utils(self):
        """Test DataLoader utils applied to NPDataLoader"""
        batch_size = 1
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size)
        expected_num_iterations = len(self.X) // batch_size
        for i, _ in enumerate(_generate_roughly_one_epoch(dataloader), start=1):
            pass
        self.assertEqual(i, expected_num_iterations)

        batch_size = 3
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size)
        expected_num_iterations = len(self.X) // batch_size + 1
        for i, _ in enumerate(_generate_roughly_one_epoch(dataloader), start=1):
            pass
        self.assertEqual(i, expected_num_iterations)

        batch_size = 3
        dataloader = NPDataLoader(dataset=self.X,
                                  target=self.X,
                                  batch_size=batch_size,
                                  drop_last=True)
        expected_num_iterations = len(self.X) // batch_size
        for i, _ in enumerate(_generate_roughly_one_epoch(dataloader), start=1):
            pass
        self.assertEqual(i, expected_num_iterations)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
