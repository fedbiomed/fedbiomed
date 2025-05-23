import unittest
import numpy as np
import pandas as pd

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

    def test_data_manager_01_load(self):

        """ Testing __getattr__ method of DataManager """

        # Test passing invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager = DataManager(dataset='invalid-argument')
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        # Test passing another invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            DataManager(dataset=12)
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
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        self.assertIsInstance(data_manager._data_manager_instance, TorchDataManager)

        # Test SkLearn Scenario
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 2]))
        data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        self.assertIsInstance(data_manager._data_manager_instance, SkLearnDataManager)

        # Test auto PyTorch dataset creation
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 2]))
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        self.assertIsInstance(data_manager._data_manager_instance, TorchDataManager)

        # Test if inputs are not supported by SkLearnTrainingPlan
        data_manager = DataManager(dataset=['non-pd-or-numpy'], target=['non-pd-or-numpy'])
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)

        # Test undefined training plan
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 2]))
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager.load(tp_type='NanaNone')

    def test_data_manager_02___getattr___(self):
        """ Test __getattr__ magic method of DataManager """

        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 2]))
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        try:
            load = data_manager.__getattr__('load')
            dataset = data_manager.__getattr__('dataset')
        except Exception as e:
            self.assertTrue(False, f'Error while calling __getattr__ method of DataManager {str(e)}')

        # Test attribute error tyr/catch block
        with self.assertRaises(FedbiomedDataManagerError):
            data_manager.__getattr__('toto')

    def test_data_manager_03_extend_loader_args(self):
        """Test that extend loader args respects the precedence rules."""
        dm_keyword_args = {'dm_keyword_argument': 'keyword_argument_data_manager'}
        data_manager = DataManager(dataset=pd.DataFrame([[1], [1]]), target=pd.Series([1, 2]),
                                   **dm_keyword_args)
        self.assertDictEqual(data_manager._loader_arguments, dm_keyword_args)
        extension_keyword_args = {'new_arg': 'should exist',
                                  'dm_keyword_argument': 'should not be changed'}
        data_manager.extend_loader_args(extension_keyword_args)
        self.assertDictEqual(data_manager._loader_arguments, {
            **extension_keyword_args,
            **dm_keyword_args
        })

    def test_data_manager_04_testing_index_setter_getter(self):
        # for sklearn
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 1]))
        data_manager.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        data_manager.testing_index = [1]
        data_manager.training_index = [0]

        train, test = data_manager.split(test_ratio=.5, test_batch_size=None)
        self.assertEqual(data_manager.testing_index, [1])
        #self.assertListEqual(data_manager.testing_index, data_manager._testing_index)

        # for pytorch
        data_manager = DataManager(dataset=pd.DataFrame([[1, 2, 3], [1, 2, 3]]), target=pd.Series([1, 1]))
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        data_manager.testing_index = [1]
        data_manager.training_index = [0]

        train, test = data_manager.split(test_ratio=.5, test_batch_size=None)
        self.assertEqual(data_manager.testing_index, [1])
        #self.assertListEqual(data_manager.testing_index, data_manager._testing_index)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
