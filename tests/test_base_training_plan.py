import unittest
from unittest.mock import patch, MagicMock
from typing import Any, Dict, Optional
import logging
import torch
import numpy as np

from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.constants import ProcessTypes
from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan  # noqa
# Import again the full module: we need it to test saving code without dependencies. Do not delete the line below.
import fedbiomed.common.training_plans._base_training_plan  # noqa


class SimpleTrainingPlan(BaseTrainingPlan):
    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
    ) -> None:
        pass

    def predict(
            self,
            data: Any,
    ) -> np.ndarray:
        pass


class TestBaseTrainingPlan(unittest.TestCase):
    """ Test Class for Base Training Plan """

    def setUp(self):
        self.tp = SimpleTrainingPlan()
        logging.disable('CRITICAL')
        pass

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_base_training_plan_01_add_dependency(self):
        """ Test  adding dependencies """

        expected = ['from torch import nn']
        self.tp.add_dependency(expected)
        self.assertListEqual(expected, self.tp._dependencies, 'Can not set dependency properly')

    def test_base_training_plan_02_set_dataset_path(self):
        """ Test setting dataset path """

        expected = '/path/to/my/data.csv'
        self.tp.set_dataset_path(expected)
        self.assertEqual(expected, self.tp.dataset_path, 'Can not set `dataset_path` properly')

    def test_base_training_plan_03_save_code(self):
        """ Testing the method save_code of BaseTrainingPlan """
        expected_filepath = 'path/to/model.py'

        # Test without dependencies
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w")

        # Test with adding dependencies
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
            self.tp.add_dependency(['from fedbiomed.common.training_plans import TorchTrainingPlan'])
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w")

        # Test if get_class_source raises error
        with patch('fedbiomed.common.training_plans._base_training_plan.get_class_source') \
                as mock_get_class_source:
            mock_get_class_source.side_effect = FedbiomedError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

        # Test if open function raises errors
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
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
            self.tp.add_preprocess(method, 'WorngType')

        # Test raising error due to wrong type of method argument
        with self.assertRaises(FedbiomedTrainingPlanError):
            self.tp.add_preprocess('not-callable', ProcessTypes.DATA_LOADER)

        # Test proper scenario
        self.tp.add_preprocess(method, ProcessTypes.DATA_LOADER)
        self.assertTrue('method' in self.tp.pre_processes, 'add_preprocess could not add process properly')

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
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom': 14})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = True
            result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = 'True'
            BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        metric = [14, 14, 14.5]
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom_1': 14, 'Custom_2': 14, 'Custom_3': 14.5})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = ['14', '14', '14']
            result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        metric = {'my_metric': 12, 'other_metric': 14.15}
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, metric)

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {'my_metric': 'True', 'other_metric': 14.15}
            result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        # Testing torch.tensor
        metric = torch.tensor(14)
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom': 14})

        metric = [torch.tensor(14), torch.tensor(14), torch.tensor(14)]
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom_1': 14, 'Custom_2': 14, 'Custom_3': 14})

        metric = {"m1": torch.tensor(14), "m2": torch.tensor(14)}
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'m1': 14, 'm2': 14})

        metric = {"m1": torch.tensor(14.5), "m2": torch.tensor(14.5)}
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'m1': 14.5, 'm2': 14.5})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {"m1": torch.tensor([14.5, 14.5]), "m2": torch.tensor([14.5,14.5])}
            BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        # Testing numpy arrays
        metric = np.array([14, 14, 14])
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom_1': 14, 'Custom_2': 14, 'Custom_3': 14})

        metric = np.array([14.5, 14.5, 14.5])
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom_1': 14.5, 'Custom_2': 14.5, 'Custom_3': 14.5})

        metric = np.array([14.5, 14.5, 14.5], dtype=np.floating)
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom_1': 14.5, 'Custom_2': 14.5, 'Custom_3': 14.5})

        with self.assertRaises(FedbiomedTrainingPlanError):
            metric = {"m1": np.array([14.5, 14.5]), "m2": np.array([14.5, 14.5])}
            BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')

        metric = np.float64(4.5)
        result = BaseTrainingPlan._create_metric_result_dict(metric=metric, metric_name='Custom')
        self.assertDictEqual(result, {'Custom': 4.5})
        self.assertIsInstance(result["Custom"], float)

    def test_base_training_plan_07_training_data(self):
        """ Test training_data method whether raises error """

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
        data = {
            'T1': MagicMock(spec=torch.Tensor)
        }
        data['T1'].__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Test complex case: data is a list of dicts of arrays
        data = [{
            'T1': MagicMock(spec=np.ndarray)
        },
            {
            'T1': MagicMock(spec=np.ndarray)
        }, ]
        data[0]['T1'].__len__.return_value = batch_size
        self.assertEqual(tp._infer_batch_size(data), batch_size)

        # Test random arbitrarily-nested data
        collection_types = ('list', 'tuple', 'dict')
        leaf_types = (torch.Tensor, np.ndarray)
        data_leaf_type = leaf_types[np.random.randint(low=0, high=len(leaf_types)-1, size=1)[0]]
        data = MagicMock(spec=data_leaf_type)
        data.__len__.return_value = batch_size
        num_nesting_levels = np.random.randint(low=1, high=5, size=1)[0]
        nesting_types = list()  # for record-keeping purposes
        for _ in range(num_nesting_levels):
            collection_type = collection_types[np.random.randint(low=0, high=len(collection_types)-1, size=1)[0]]
            nesting_types.append(collection_type)
            if collection_type == 'list':
                data = [data, data]
            elif collection_type == 'tuple':
                data = (data, data)
            else:
                data = {'T1': data, 'T2': data}
        self.assertEqual(tp._infer_batch_size(data), batch_size,
                         f'Inferring batch size failed on arbitrary nested collection of {nesting_types[::-1]} '
                         f'and leaf type {data_leaf_type.__name__}')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
