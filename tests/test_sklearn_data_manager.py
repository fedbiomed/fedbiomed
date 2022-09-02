import math
import unittest
import testsupport.mock_node_environ  # noqa (remove flake8 false warning)
import numpy as np
import pandas as pd

from fedbiomed.common.data._sklearn_data_manager import SkLearnDataManager
from fedbiomed.common.exceptions import FedbiomedSkLearnDataManagerError


class TestSkLearnDataManager(unittest.TestCase):

    def setUp(self):
        # Setup global TorchDataset class
        self.inputs = np.array([[1, 4, 3, 7],
                                [4, 6, 3, 1],
                                [1, 5, 3, 7],
                                [8, 2, 6, 9]
                                ])
        self.target = np.array([5, 5, 1, 4])
        self.sklearn_data_manager = SkLearnDataManager(inputs=self.inputs,
                                                       target=self.target)

    def tearDown(self):
        pass

    def test_sklearn_data_manager_01_init(self):
        """ Testing dataset getter method """

        # Test if arguments provided as pd.DataFrame and they have been properly converted to the
        # np.ndarray
        inputs = pd.DataFrame(self.inputs)
        target = pd.DataFrame(self.target)
        self.sklearn_data_manager = SkLearnDataManager(inputs=inputs,
                                                       target=target)
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
        with self.assertRaises(FedbiomedSkLearnDataManagerError):
            self.sklearn_data_manager.split(test_ratio=-1)

        with self.assertRaises(FedbiomedSkLearnDataManagerError):
            self.sklearn_data_manager.split(test_ratio=2)

        with self.assertRaises(FedbiomedSkLearnDataManagerError):
            self.sklearn_data_manager.split(test_ratio='not-int-or-float')

        # Get number of samples
        n_samples = len(self.sklearn_data_manager.dataset()[0])

        ratio = 0.5
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test
        loader_train, loader_test = self.sklearn_data_manager.split(test_ratio=ratio)

        self.assertEqual(len(loader_test[0]), n_test, 'Number of samples of test loader is not as expected')
        self.assertEqual(len(loader_test[1]), n_test, 'Number of samples of test loader is not as expected')
        self.assertEqual(len(loader_train[0]), n_train, 'Number of samples of train loader is not as expected')
        self.assertEqual(len(loader_train[1]), n_train, 'Number of samples of train loader is not as expected')

        loader_all = self.sklearn_data_manager.load_all_samples()
        self.assertEqual(len(loader_all[0]), n_samples, 'Number of samples of the loader that is returned by '
                                                        '`load_all_samples` is not as expected')
        self.assertEqual(len(loader_all[1]), n_samples, 'Number of samples of the loader that is returned by '
                                                        '`load_all_samples` is not as expected')


        # Test if test ratio is 1
        ratio = 1
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test
        loader_train, loader_test = self.sklearn_data_manager.split(test_ratio=ratio)
        self.assertEqual(len(loader_test[0]), n_test, 'Number of samples of test loader is not as expected')
        self.assertEqual(len(loader_test[1]), n_test, 'Number of samples of test loader is not as expected')
        self.assertIsNone(loader_train, 'Expected train set is None')
        self.assertIsNone(loader_train, 'Expected train set is None')

        # Test if test ratio is 0
        ratio = 0
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test
        loader_train, loader_test = self.sklearn_data_manager.split(test_ratio=ratio)
        self.assertIsNone(loader_test)
        self.assertIsNone(loader_test)
        self.assertEqual(len(loader_train[0]), n_train, 'Number of samples of train loader is not as expected')
        self.assertEqual(len(loader_train[1]), n_train, 'Number of samples of train loader is not as expected')

    def test_sklearn_data_manager_03_getter_subsets(self):
        """ Test getter for subset train and subset test"""

        ratio = 0.5
        n_samples = len(self.sklearn_data_manager.dataset()[0])
        n_test = math.floor(n_samples * ratio)
        n_train = n_samples - n_test

        self.sklearn_data_manager.split(test_ratio=ratio)

        subset_test = self.sklearn_data_manager.subset_test()
        subset_train = self.sklearn_data_manager.subset_train()
        self.assertEqual(len(subset_test[0]), n_test)
        self.assertEqual(len(subset_test[1]), n_test)
        self.assertEqual(len(subset_train[0]), n_train)
        self.assertEqual(len(subset_train[1]), n_train)

    def test_sklearn_data_manager_04_subset_loader(self):

        # Invalid subset
        with self.assertRaises(FedbiomedSkLearnDataManagerError):
            self.sklearn_data_manager._subset_loader(subset=np.array([1, 2, 3]))

        # Invalid nested subset
        with self.assertRaises(FedbiomedSkLearnDataManagerError):
            self.sklearn_data_manager._subset_loader(subset=([1, 2, 3], [1, 2, 3]))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
