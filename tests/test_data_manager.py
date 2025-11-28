import unittest

import numpy as np
import pandas as pd

from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.datamanager import (
    DataManager,
    TorchDataManager,
)
from fedbiomed.common.datamanager._sklearn_data_manager import SkLearnDataManager
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset._native_dataset import NativeDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError


class TestDataManager(unittest.TestCase):
    class CustomDataset(Dataset):
        """Create Fed-BioMed Dataset for test purposes"""

        def __init__(self):
            self.X_train = []
            self.Y_train = []

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

        def to_format(self) -> DataReturnFormat:
            return DataReturnFormat.TORCH

        def complete_initialization(self) -> None:
            pass

        def complete_initialization(self):
            return None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # NOTE: alitolga: Which data types we support for NativeDataset? Depending on that some cases in this test become invalid.
    def test_dataset_numpy(self, n=8, d=3):
        X = np.arange(n * d, dtype=float).reshape(n, d)
        y = (np.arange(n) % 2).astype(int)
        return X, y

    def test_dataset_pandas(self, n=8, d=3):
        X, y = self.test_dataset_numpy(n=n, d=d)
        return pd.DataFrame(X), pd.Series(y)

    def test_data_manager_01_load(self):
        """Testing __getattr__ method of DataManager"""

        # Test passing invalid argument
        with self.assertRaises(FedbiomedError):
            data_manager = DataManager(dataset="invalid-argument")
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
            data_manager.complete_dataset_initialization(
                controller_kwargs={"root": "dummy_path"}
            )

        # Test passing another invalid argument
        with self.assertRaises(FedbiomedError):
            DataManager(dataset=12)
            data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
            data_manager.complete_dataset_initialization(
                controller_kwargs={"root": "dummy_path"}
            )

        # Test Native Dataset Scenario
        data_manager = DataManager(dataset=[12, 12, 12, 12])
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        data_manager.complete_dataset_initialization(
            controller_kwargs={"root": "dummy_path"}
        )
        self.assertIsInstance(data_manager._data_manager_instance, TorchDataManager)
        self.assertIsInstance(data_manager._dataset, NativeDataset)

        # Test Torch Dataset Scenario
        data_manager = DataManager(dataset=TestDataManager.CustomDataset())
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        self.assertIsInstance(data_manager._data_manager_instance, TorchDataManager)

        # sklearn route — NumPy (X, y)
        X, y = self.test_dataset_numpy(n=6, d=2)
        dm = DataManager(dataset=X, target=y)
        dm.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        self.assertIsInstance(dm._data_manager_instance, SkLearnDataManager)
        self.assertEqual(
            dm._data_manager_instance.dataset.to_format, DataReturnFormat.SKLEARN
        )

        # sklearn route — NumPy with target=None
        X2, _ = self.test_dataset_numpy(n=5, d=2)
        dm2 = DataManager(dataset=X2, target=None)
        dm2.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        self.assertIsInstance(dm2._data_manager_instance, SkLearnDataManager)

        # sklearn route — pandas DataFrame/Series
        Xp, yp = self.test_dataset_pandas(n=4, d=2)
        dmp = DataManager(dataset=Xp, target=yp)
        dmp.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        self.assertIsInstance(dmp._data_manager_instance, SkLearnDataManager)

        # Test undefined training plan
        data_manager = DataManager(dataset=TestDataManager.CustomDataset())
        with self.assertRaises(FedbiomedError):
            data_manager.load(tp_type="NanaNone")

    def test_data_manager_02___getattr___(self):
        """Test __getattr__ magic method of DataManager"""

        data_manager = DataManager(dataset=TestDataManager.CustomDataset())
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
        try:
            data_manager.__getattr__("load")
            data_manager.__getattr__("dataset")
        except Exception as e:
            self.assertTrue(
                False, f"Error while calling __getattr__ method of DataManager {str(e)}"
            )

        # Test attribute error tyr/catch block
        with self.assertRaises(FedbiomedError):
            data_manager.__getattr__("toto")

        # sklearn path
        X, y = self.test_dataset_numpy(n=6, d=2)
        dm_sk = DataManager(dataset=X, target=y)
        dm_sk.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        try:
            dm_sk.__getattr__("load")
            dm_sk.__getattr__("dataset")
        except Exception as e:
            self.fail(f"__getattr__ failed on sklearn path: {e}")

    def test_data_manager_03_extend_loader_args(self):
        """Test that extend loader args respects the precedence rules."""
        dm_keyword_args = {"dm_keyword_argument": "keyword_argument_data_manager"}
        data_manager = DataManager(
            dataset=TestDataManager.CustomDataset(),
            **dm_keyword_args,
        )
        self.assertDictEqual(data_manager._loader_arguments, dm_keyword_args)
        extension_keyword_args = {
            "new_arg": "should exist",
            "dm_keyword_argument": "should not be changed",
        }
        data_manager.extend_loader_args(extension_keyword_args)
        self.assertDictEqual(
            data_manager._loader_arguments,
            {**extension_keyword_args, **dm_keyword_args},
        )

        # NOTE: @alitolga - is a similar test required here for sklearn data manager?
        # Does the SkLearnDataLoader need to take kwargs or not?

    def test_data_manager_04_testing_index_setter_getter(self):
        # sklearn path — NumPy (tuple input)
        X, y = self.test_dataset_numpy(n=9, d=2)
        dm_s = DataManager(dataset=(X, y))
        dm_s.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        dm_s._testing_index = [2, 4]
        dm_s._training_index = [
            i for i in range(len(X)) if i not in dm_s._testing_index
        ]
        tr_s, te_s = dm_s.split(
            test_ratio=len(dm_s._testing_index) / len(X), test_batch_size=None
        )
        self.assertListEqual(dm_s._testing_index, [2, 4])

        # sklearn path — edge ratios
        X2, _ = self.test_dataset_numpy(n=4, d=2)
        dm_none = DataManager(dataset=X2, target=None)
        dm_none.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        tr0, ts0 = dm_none.split(test_ratio=0.0, test_batch_size=None)
        self.assertIsNotNone(tr0)
        self.assertIsNone(ts0)
        tr1, ts1 = dm_none.split(test_ratio=1.0, test_batch_size=None)
        self.assertIsNone(tr1)
        self.assertIsNotNone(ts1)

        # pandas variant
        Xp, yp = self.test_dataset_pandas(n=7, d=2)
        dm_p = DataManager(dataset=Xp, target=yp)
        dm_p.load(tp_type=TrainingPlans.SkLearnTrainingPlan)
        tr_p, ts_p = dm_p.split(test_ratio=0.3, test_batch_size=None)
        self.assertTrue(tr_p is not None or ts_p is not None)

        # for pytorch
        data_manager = DataManager(dataset=TestDataManager.CustomDataset())
        data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)

        data_manager._testing_index = [1]
        data_manager._training_index = [0]

        train, test = data_manager.split(test_ratio=0.5, test_batch_size=None)
        self.assertEqual(data_manager._testing_index, [1])
        # self.assertListEqual(data_manager._testing_index, data_manager._testing_index)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
