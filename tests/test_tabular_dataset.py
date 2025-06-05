import unittest
import pandas as pd
import numpy as np
import torch

from fedbiomed.common.data import TabularDataset
from fedbiomed.common.exceptions import FedbiomedDatasetError


class TestTabularDataset(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_torch_data_manager_01_initialization(self):
        """ Testing TabularDataset initialization scenarios"""

        # Test if inputs is not in proper type
        with self.assertRaises(FedbiomedDatasetError):
            TabularDataset(inputs=[1, 2, 3], target=pd.Series([1, 2]))

        # Test if target argument is not in proper type
        with self.assertRaises(FedbiomedDatasetError):
            TabularDataset(inputs=pd.Series([1, 2]), target='toto')

        # Test if input and target pd.DataFrame o
        inputs = pd.DataFrame([[1.2, 2, 3],
                               [0.4, 5, 4],
                               [0, 2, 4]])
        dataset = TabularDataset(inputs=inputs, target=pd.DataFrame([1, 2, 3]))
        self.assertIsInstance(dataset.inputs, torch.Tensor)
        self.assertIsInstance(dataset.target, torch.Tensor)

        # Test if input and target are pd.Series
        inputs = pd.Series([1.2, 2, 3])
        dataset = TabularDataset(inputs=inputs, target=pd.Series([1, 2, 3]))
        self.assertIsInstance(dataset.inputs, torch.Tensor)
        self.assertIsInstance(dataset.target, torch.Tensor)

        # Test if target and inputs are numpy array
        inputs = np.array([[1, 2, 3],
                           [1, 2, 4],
                           [1, 2, 4]])

        dataset = TabularDataset(inputs=inputs, target=np.array([1, 2, 3]))
        self.assertIsInstance(dataset.inputs, torch.Tensor)
        self.assertIsInstance(dataset.target, torch.Tensor)

        # Test the scenario where number of samples do not match
        with self.assertRaises(FedbiomedDatasetError):
            TabularDataset(inputs=inputs, target=pd.Series([1, 2]))

    def test_torch_data_manager_02_magic_len(self):
        """Testing magic method __len__ of TorchTabular dataset"""

        inputs = np.array([[1, 2, 3],
                           [1, 2, 4],
                           [1, 2, 4]])
        dataset = TabularDataset(inputs=inputs, target=np.array([1, 2, 3]))
        leng = dataset.__len__()
        self.assertEqual(leng, 3)

    def test_torch_data_manager_02_magic_getitem(self):
        """Testing magic method __len__ of TorchTabular dataset"""

        inputs = np.array([[1, 2, 3],
                           [5, 2, 4],
                           [1, 2, 4]])
        dataset = TabularDataset(inputs=inputs, target=np.array([1, 2, 3]))

        # Should return tuple (tensor([5., 2., 4.]), tensor(2.))
        row = dataset.__getitem__(1)
        self.assertIsInstance(row, tuple)
        self.assertEqual(row[0][0].item(), 5.0, 'Get item does not return correct value in inputs')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
