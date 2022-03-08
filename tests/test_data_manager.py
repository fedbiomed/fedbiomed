import unittest

import numpy as np
from unittest.mock import patch, MagicMock

import pandas as pd

import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

from torch.utils.data import Dataset
from fedbiomed.common.data import DataManager
from fedbiomed.common.data._torch_data_manager import TorchDataManager
from fedbiomed.common.data._sklearn_data_manager import SkLearnDataManager
from fedbiomed.common.exceptions import FedbiomedDataManagerError
from fedbiomed.common.constants import TrainingPlans

class TestDataManager(unittest.TestCase):

    class CustomDataset(Dataset):
        """ Create PyTorch Dataset for test purposes """
        def __init__(self):
            self.X_train = []
            self.Y_train = []

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager_01_initialization(self):
        """ Testing different initializations of DataManger """

        pass

    def test_data_manager_02_load(self):

        """ Testing __getattr__ method of DataManager """

        # Test passing invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager = DataManager(dataset='invalid-argument')
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        # Test passing another invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager = DataManager(dataset=12)
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        # Test passing dataset as list
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager = DataManager(dataset=[12, 12, 12, 12])
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        # Test passing PyTorch Dataset while training plan is SkLearn
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager = DataManager(dataset=TestDataManager.CustomDataset())
            data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)

        # Test Torch Dataset Scenario
        data_manager = DataManager(dataset=TestDataManager.CustomDataset())
        data_manager = data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        self.assertIsInstance(data_manager, TorchDataManager)

        # Test SkLearn Scenario
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 2, 3]))
        data_manager = data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        self.assertIsInstance(data_manager, SkLearnDataManager)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
