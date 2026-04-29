import logging
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Import again the full module: we need it to test saving code without dependencies. Do not delete the line below.
import fedbiomed.common.training_plans._base_training_plan  # noqa
from fedbiomed.common.constants import ProcessTypes
from fedbiomed.common.dataloader import SkLearnDataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.models._torch import TorchModel
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan  # noqa
from fedbiomed.node.history_monitor import HistoryMonitor


class SimpleTrainingPlan(BaseTrainingPlan):
    def training_routine(
        self,
        history_monitor: Optional["HistoryMonitor"] = None,
        node_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def post_init(
        self, model_args: Dict[str, Any], training_args: Dict[str, Any]
    ) -> None:
        pass

    def model(self):
        pass

    def predict(
        self,
        data: Any,
    ) -> np.ndarray:
        pass

    def init_optimizer(self):
        pass


class TestBaseTrainingPlan(unittest.TestCase):
    """Test Class for Base Training Plan"""

    def setUp(self):
        self.tp = SimpleTrainingPlan()
        logging.disable("CRITICAL")
        pass

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_base_training_plan_01__add_dependency(self):
        """Test  adding dependencies"""

        expected = ["from torch import nn"]
        self.tp._add_dependency(expected)
        self.assertListEqual(
            expected, self.tp._dependencies, "Can not set dependency properly"
        )

    def test_base_training_plan_02_set_dataset_path(self):
        """Test setting dataset path"""

        expected = "/path/to/my/data.csv"
        self.tp.set_dataset_path(expected)
        self.assertEqual(
            expected, self.tp.dataset_path, "Can not set `dataset_path` properly"
        )

    def test_base_training_plan_03_save_code(self):
        """Testing the method save_code of BaseTrainingPlan"""
        expected_filepath = "path/to/model.py"

        # Test without dependencies
        with patch.object(
            fedbiomed.common.training_plans._base_training_plan, "open", MagicMock()
        ) as mock_open:
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")

        # Test with adding dependencies
        with patch.object(
            fedbiomed.common.training_plans._base_training_plan, "open", MagicMock()
        ) as mock_open:
            self.tp._add_dependency(
                ["from fedbiomed.common.training_plans import TorchTrainingPlan"]
            )
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")

        # Test if get_class_source raises error
        with patch(
            "fedbiomed.common.training_plans._base_training_plan.get_class_source"
        ) as mock_get_class_source:
            mock_get_class_source.side_effect = FedbiomedError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

        # Test if open function raises errors
        with patch.object(
            fedbiomed.common.training_plans._base_training_plan, "open", MagicMock()
        ) as mock_open:
            mock_open.side_effect = OSError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

            mock_open.side_effect = PermissionError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

            mock_open.side_effect = MemoryError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

    def test_base_training_plan_04_add_preprocess(self):
        def method(args):
            pass

        # Test raising error due to worn process type
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.add_preprocess(method, "WorngType")

        # Test raising error due to wrong type of method argument
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.add_preprocess("not-callable", ProcessTypes.DATA_LOADER)

        # Test proper scenario
        self.tp.add_preprocess(method, ProcessTypes.DATA_LOADER)
        self.assertTrue(
            "method" in self.tp.pre_processes,
            "add_preprocess could not add process properly",
        )

    def test_base_training_plan_05_set_data_loaders(self):
        test_data_loader = [1, 2, 3]
        train_data_loader = [1, 2, 3]

        self.tp.set_data_loaders(train_data_loader, test_data_loader)
        self.assertListEqual(self.tp.training_data_loader, train_data_loader)
        self.assertListEqual(self.tp.testing_data_loader, test_data_loader)

    def test_base_training_plan_06__create_metric_result(self):
        """
        Testing private method create metric result dict

        This test function also tests the method _check_metric_types_is_int_or_float
        as implicitly
        """

        metric = 14
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom": 14})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = True
            result = BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = "True"
            BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        metric = [14, 14, 14.5]
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom_1": 14, "Custom_2": 14, "Custom_3": 14.5})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = ["14", "14", "14"]
            result = BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        metric = {"my_metric": 12, "other_metric": 14.15}
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, metric)

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {"my_metric": "True", "other_metric": 14.15}
            result = BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        # Testing torch.tensor
        metric = torch.tensor(14)
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom": 14})

        metric = [torch.tensor(14), torch.tensor(14), torch.tensor(14)]
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom_1": 14, "Custom_2": 14, "Custom_3": 14})

        metric = {"m1": torch.tensor(14), "m2": torch.tensor(14)}
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"m1": 14, "m2": 14})

        metric = {"m1": torch.tensor(14.5), "m2": torch.tensor(14.5)}
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"m1": 14.5, "m2": 14.5})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {
                "m1": torch.tensor([14.5, 14.5]),
                "m2": torch.tensor([14.5, 14.5]),
            }
            BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        # Testing numpy arrays
        metric = np.array([14, 14, 14])
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom_1": 14, "Custom_2": 14, "Custom_3": 14})

        metric = np.array([14.5, 14.5, 14.5])
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(
            result, {"Custom_1": 14.5, "Custom_2": 14.5, "Custom_3": 14.5}
        )

        metric = np.array([14.5, 14.5, 14.5], dtype=np.floating)
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(
            result, {"Custom_1": 14.5, "Custom_2": 14.5, "Custom_3": 14.5}
        )

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {"m1": np.array([14.5, 14.5]), "m2": np.array([14.5, 14.5])}
            BaseTrainingPlan._create_metric_result_dict(
                metric=metric, metric_name="Custom"
            )

        metric = np.float64(4.5)
        result = BaseTrainingPlan._create_metric_result_dict(
            metric=metric, metric_name="Custom"
        )
        self.assertDictEqual(result, {"Custom": 4.5})
        self.assertIsInstance(result["Custom"], float)

    def test_base_training_plan_07_training_data(self):
        """Test training_data method whether raises error"""

        # The method training data should be defined by user, that's why
        # training_data in BaseTrainingPLan has been configured for raising error
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.training_data()

    def test_base_training_plan_08_get_learning_rate(self):
        pass

    def test_base_training_plan_09_infer_batch_size(self):
        """Test that the utility to infer batch size works correctly.

        Supported data types are:
            - torch tensor
            - numpy array
            - dict mapping {modality: tensor/array}
            - tuple or list containing the above
        """
        batch_size = 4
        tp = SimpleTrainingPlan()

        # Test simple case: data is a tensor
        data = MagicMock(spec=torch.Tensor)
        data.__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Test simple case: data is an array
        data = MagicMock(spec=np.ndarray)
        data.__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Text complex case: data is a dict of tensors
        data = {"T1": MagicMock(spec=torch.Tensor)}
        data["T1"].__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Test complex case: data is a list of dicts of arrays
        data = [
            {"T1": MagicMock(spec=np.ndarray)},
            {"T1": MagicMock(spec=np.ndarray)},
        ]
        data[0]["T1"].__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Test random arbitrarily-nested data
        collection_types = ("list", "tuple", "dict")
        leaf_types = (torch.Tensor, np.ndarray)
        data_leaf_type = leaf_types[
            np.random.randint(low=0, high=len(leaf_types) - 1, size=1)[0]
        ]
        data = MagicMock(spec=data_leaf_type)
        data.__len__.return_value = batch_size
        num_nesting_levels = np.random.randint(low=1, high=5, size=1)[0]
        nesting_types = list()  # for record-keeping purposes
        for _ in range(num_nesting_levels):
            collection_type = collection_types[
                np.random.randint(low=0, high=len(collection_types) - 1, size=1)[0]
            ]
            nesting_types.append(collection_type)
            if collection_type == "list":
                data = [data, data]
            elif collection_type == "tuple":
                data = (data, data)
            else:
                data = {"T1": data, "T2": data}
        self.assertEqual(
            tp._infer_batch_size(data),
            batch_size,
            f"Inferring batch size failed on arbitrary nested collection of {nesting_types[::-1]} "
            f"and leaf type {data_leaf_type.__name__}",
        )

    # test for bug #893
    def test_base_training_plan_10_node_out_of_memory_bug(self):
        # bug description: validation testing uses batch size equal to all the validation dataset,
        # which leads to OutOfMemory errors. Tests that testing_routine iterates over
        # test_batch_size-sized batches rather than loading the whole dataset at once.
        # Covers both sklearn (SkLearnDataLoader) and pytorch (torch.utils.data.DataLoader) paths.

        class FakeSkLearnDataset(Dataset):
            def __init__(self, nb_samples: int, n_features: int = 10):
                self._data = [np.random.randn(n_features) for _ in range(nb_samples)]
                self._target = [
                    np.array(np.random.randint(0, 2)) for _ in range(nb_samples)
                ]

            def complete_initialization(self, *args, **kwargs) -> None:
                pass

            def __getitem__(self, idx: int):
                return self._data[idx], self._target[idx]

            def __len__(self) -> int:
                return len(self._data)

        class FakeTorchDataset(torch.utils.data.Dataset):
            def __init__(self, nb_samples: int = 1234):
                self.data = torch.randn(nb_samples, 100)
                self.label = torch.randint(0, 5, size=(nb_samples,))
                self.nb_samples = nb_samples

            def __len__(self):
                return self.nb_samples

            def __getitem__(self, idx):
                return self.data[idx], self.label[idx]

        nb_samples_test_dataset = 123
        test_batch_sizes = (
            1,
            2,
            10,
            nb_samples_test_dataset,
            nb_samples_test_dataset + 10,
            nb_samples_test_dataset - 1,
        )

        loader_cases = [
            # (train_loader, test_loader_factory, predict_fn, model_spec)
            (
                SkLearnDataLoader(dataset=FakeSkLearnDataset(64)),
                lambda bs: SkLearnDataLoader(
                    dataset=FakeSkLearnDataset(nb_samples_test_dataset), batch_size=bs
                ),
                lambda x: np.zeros(x.shape[0], dtype=int),
                None,
            ),
            (
                torch.utils.data.DataLoader(FakeTorchDataset(1234)),
                lambda bs: torch.utils.data.DataLoader(
                    FakeTorchDataset(nb_samples_test_dataset), batch_size=bs
                ),
                lambda x: x.T[0].numpy(),
                TorchModel,
            ),
        ]

        for train_loader, test_loader_factory, predict_fn, model_spec in loader_cases:
            for test_batch_size in test_batch_sizes:
                model = MagicMock(
                    predict=MagicMock(side_effect=predict_fn),
                    **({"spec": model_spec} if model_spec else {}),
                )
                self.tp._model = model
                self.tp._training_args = TrainingArgs(only_required=False)
                self.tp.set_data_loaders(
                    train_loader, test_loader_factory(test_batch_size)
                )
                self.tp.testing_routine(
                    metric=None,
                    metric_args={},
                    history_monitor=None,
                    before_train=True,
                )
                nb_test_batch = nb_samples_test_dataset // test_batch_size + int(
                    nb_samples_test_dataset % test_batch_size > 0
                )
                self.assertEqual(model.predict.call_count, nb_test_batch)

    def test_base_training_plan_11_local_params_property(self):
        """Test local_params property"""

        self.assertIsNone(self.tp.local_params)

        self.tp._local_params = ["param1", "param2"]
        self.assertEqual(self.tp.local_params, ["param1", "param2"])

    def test_base_training_plan_12_get_dp_controller(self):
        """Test default DP controller getter"""
        self.assertIsNone(self.tp.get_dp_controller())

    def test_base_training_plan_13_validate_parameter_tags(self):
        """Test validation of parameter tags"""

        # valid case
        tags = {"local"}
        validated = self.tp._validate_parameter_tags("weight", tags)
        self.assertEqual(validated, tags)

        # invalid type
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp._validate_parameter_tags("weight", ["local"])

        # invalid tag
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp._validate_parameter_tags("weight", {"invalid"})

    def test_base_training_plan_14_get_parameter_tags(self):
        """Test retrieving parameter tags"""

        tags = self.tp.get_parameter_tags("weight")
        self.assertEqual(tags, set())

    def test_base_training_plan_15_filter_model_params_by_tags(self):
        """Test filtering model parameters based on tags"""

        params = {"w1": 1, "w2": 2, "w3": 3}

        def fake_tag_parameters(name):
            if name == "w1":
                return {"local"}
            if name == "w2":
                return {"persistent"}
            return set()

        self.tp.tag_parameters = fake_tag_parameters

        # filter required tag
        filtered = self.tp.filter_model_params_by_tags(params, required_tags={"local"})
        self.assertEqual(filtered, {"w1": 1})
        # invalid required tag
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.filter_model_params_by_tags(params, required_tags=["local"])

        # filter forbidden tag
        filtered = self.tp.filter_model_params_by_tags(params, forbidden_tags={"local"})
        self.assertNotIn("w1", filtered)
        # invalid forbidden tag
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.filter_model_params_by_tags(params, forbidden_tags="local")

    def test_base_training_plan_16_get_model_params_local(self):
        """Test get_model_params passes local_params to model"""

        model = MagicMock()
        model.get_weights.return_value = {"w": 1}

        self.tp._model = model

        self.tp.get_model_params(local_params=["w"])

        model.get_weights.assert_called_once_with(
            only_trainable=False, exclude_buffers=True, local_params=["w"]
        )

    def test_base_training_plan_17_set_model_params_local(self):
        """Test set_model_params passes local_params to model"""

        model = MagicMock()
        self.tp._model = model

        params = {"w": 1}

        self.tp.set_model_params(params, local_params=["w"])

        model.set_weights.assert_called_once_with(params, local_params=["w"])

    def test_base_training_plan_18_after_training_params(self):
        """Test after_training_params uses local_params"""

        model = MagicMock()
        model.get_weights.return_value = {"w": 1}

        self.tp._model = model
        self.tp._local_params = ["w"]
        self.tp._training_args = {"share_persistent_buffers": False}

        self.tp.after_training_params()

        model.get_weights.assert_called_once_with(
            only_trainable=False, exclude_buffers=True, local_params=["w"]
        )

    def test_base_training_plan_19_import_model(self):
        """Test import_model passes local_params"""

        model = MagicMock()
        self.tp._model = model
        self.tp._local_params = ["w"]

        self.tp.import_model("model.pt")

        model.reload.assert_called_once_with("model.pt", ["w"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
