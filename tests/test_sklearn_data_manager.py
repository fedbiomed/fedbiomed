import math
import unittest

import numpy as np
import pandas as pd

from fedbiomed.common.datamanager import SkLearnDataManager
from fedbiomed.common.exceptions import FedbiomedTypeError


class TestSkLearnDataManager(unittest.TestCase):
    def setUp(self):
        # Setup global TorchDataset class
        self.inputs = np.array([[1, 4, 3, 7], [4, 6, 3, 1], [1, 5, 3, 7], [8, 2, 6, 9]])
        self.target = np.array([5, 5, 1, 4])
        self.sklearn_data_manager = SkLearnDataManager(
            inputs=self.inputs, target=self.target
        )

    def assertIterableEqual(self, it1, it2):
        self.assertListEqual([x for x in it1], [x for x in it2])

    def assertNPArrayEqual(self, arr1, arr2):
        self.assertIterableEqual(arr1.flatten(), arr2.flatten())

    def test_sklearn_data_manager_01_init(self):
        """Testing dataset getter method"""

        # Test if arguments provided as pd.DataFrame and they have been properly converted to the
        # np.ndarray
        inputs = pd.DataFrame(self.inputs)
        target = pd.DataFrame(self.target)
        self.sklearn_data_manager = SkLearnDataManager(inputs=inputs, target=target)
        self.assertIsInstance(self.sklearn_data_manager._inputs, np.ndarray)
        self.assertIsInstance(self.sklearn_data_manager._target, np.ndarray)

    def test_sklearn_data_manager_02_getter_dataset(self):
        result = self.sklearn_data_manager.dataset()
        self.assertTupleEqual((self.inputs, self.target), result)

    def test_sklearn_data_manager_03_split(self):
        """
        Testing split method of SkLearnDataManager
         - Test _subset_loader
        """
        with self.assertRaises(FedbiomedTypeError):
            self.sklearn_data_manager.split(test_ratio=-1.0, test_batch_size=None)

        with self.assertRaises(FedbiomedTypeError):
            self.sklearn_data_manager.split(test_ratio=2.0, test_batch_size=None)

        with self.assertRaises(FedbiomedTypeError):
            self.sklearn_data_manager.split(
                test_ratio="not-float", test_batch_size=None
            )

        # Get number of samples
        n_samples = len(self.sklearn_data_manager.dataset()[0])

        ratio = 0.5
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test
        loader_train, loader_test = self.sklearn_data_manager.split(
            test_ratio=ratio, test_batch_size=None
        )

        msg_test = "Number of samples of test loader is not as expected"
        msg_train = "Number of samples of train loader is not as expected"
        self.assertEqual(len(loader_test.dataset), n_test, msg_test)
        self.assertEqual(len(loader_train.dataset), n_train, msg_train)
        self.assertEqual(len(self.sklearn_data_manager._testing_index), 2)
        self.assertEqual(self.sklearn_data_manager._test_ratio, ratio)

        # Test if test ratio is 1
        ratio = 1.0
        loader_train, loader_test = self.sklearn_data_manager.split(
            test_ratio=ratio, test_batch_size=None
        )
        self.assertEqual(len(loader_test.dataset), n_samples, msg_test)
        self.assertEqual(len(loader_train.dataset), 0)
        self.assertListEqual(
            self.sklearn_data_manager._testing_index, list(range(len(self.inputs)))
        )

        # Test if test ratio is 0
        ratio = 0.0
        loader_train, loader_test = self.sklearn_data_manager.split(
            test_ratio=ratio, test_batch_size=None
        )
        self.assertEqual(len(loader_test.dataset), 0, msg_test)
        self.assertEqual(len(loader_train.dataset), n_samples, msg_train)
        self.assertListEqual(self.sklearn_data_manager._testing_index, [])

    def test_sklearn_data_manager_03_getter_subsets(self):
        """Test getter for subset train and subset test"""
        ratio = 0.5
        n_samples = len(self.sklearn_data_manager.dataset()[0])
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test

        self.sklearn_data_manager.split(test_ratio=ratio, test_batch_size=None)

        subset_test = self.sklearn_data_manager.subset_test()
        subset_train = self.sklearn_data_manager.subset_train()
        self.assertEqual(len(subset_test[0]), n_test)
        self.assertEqual(len(subset_test[1]), n_test)
        self.assertEqual(len(subset_train[0]), n_train)
        self.assertEqual(len(subset_train[1]), n_train)

    def test_sklearn_data_manager_04_subset_loader(self):
        # Invalid subset
        with self.assertRaises(FedbiomedTypeError):
            self.sklearn_data_manager._subset_loader(subset=np.array([1, 2, 3]))

        # Invalid nested subset
        with self.assertRaises(FedbiomedTypeError):
            self.sklearn_data_manager._subset_loader(subset=([1, 2, 3], [1, 2, 3]))

    def test_sklearn_data_manager_05_integration_with_npdataloader(self):
        test_ratio = 0.0
        sklearn_data_manager = SkLearnDataManager(
            inputs=self.inputs,
            target=self.target,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        # random_seed=1234)

        self.assertDictEqual(
            {"batch_size": 1, "shuffle": False, "drop_last": False},
            sklearn_data_manager._loader_arguments,
        )

        loader_train, loader_test = sklearn_data_manager.split(
            test_ratio=test_ratio, test_batch_size=None
        )
        self.assertEqual(len(loader_test), 0)

        for i, (data, target) in enumerate(loader_train):
            self.assertNPArrayEqual(data, self.inputs[i, :])
            self.assertNPArrayEqual(target, self.target[i])

        batch_size = 3
        sklearn_data_manager = SkLearnDataManager(
            inputs=self.inputs,
            target=self.target,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

        loader_train, loader_test = sklearn_data_manager.split(
            test_ratio=test_ratio, test_batch_size=None
        )
        self.assertEqual(len(loader_test), 0)

        count_iter = 0
        for _, (data, target) in enumerate(loader_train):
            self.assertNPArrayEqual(data, self.inputs[:batch_size, :])
            self.assertNPArrayEqual(target, self.target[:batch_size])
            count_iter += 1

        self.assertEqual(
            count_iter, 1
        )  # assert that only one iteration was made because of drop_last=True

    def test_sklearn_data_manager_06_save_load_state(self):
        init_state = self.sklearn_data_manager.save_state()
        self.assertDictContainsSubset(
            {"testing_index": [], "training_index": [], "test_ratio": None}, init_state
        )

        ratio = 0.5
        # n_samples = len(self.sklearn_data_manager.dataset()[0])
        # n_test = math.floor(n_samples * ratio)
        # n_train = n_samples - n_test

        train_loader, test_loader = self.sklearn_data_manager.split(
            test_ratio=ratio, test_batch_size=None
        )

        state = self.sklearn_data_manager.save_state()

        new_sklearn_data_manager = SkLearnDataManager(
            self.inputs,
            self.target,
            **{"random_seed": 1234, "shuffle_testing_dataset": False},
        )
        new_sklearn_data_manager.load_state(state)

        self.assertListEqual(
            self.sklearn_data_manager._testing_index,
            new_sklearn_data_manager._testing_index,
        )
        # test with same `test_ratio` as before
        new_train_loader, new_test_loader = new_sklearn_data_manager.split(
            test_ratio=ratio, test_batch_size=None
        )

        self.assertTrue(np.array_equal(train_loader.dataset, new_train_loader.dataset))
        self.assertTrue(np.array_equal(test_loader.dataset, new_test_loader.dataset))

        # check that testing dataset is re-shuffled if `test_ratio` changes
        del new_sklearn_data_manager
        new_sklearn_data_manager = SkLearnDataManager(
            self.inputs,
            self.target,
            **{"random_seed": 1234, "shuffle_testing_dataset": False},
        )
        new_sklearn_data_manager.load_state(state)

        train_loader_reshuffled, test_loader_reshuffled = (
            new_sklearn_data_manager.split(test_ratio=0.75, test_batch_size=None)
        )

        self.assertFalse(
            np.array_equal(train_loader.dataset, train_loader_reshuffled.dataset)
        )
        self.assertFalse(
            np.array_equal(test_loader.dataset, test_loader_reshuffled.dataset)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
