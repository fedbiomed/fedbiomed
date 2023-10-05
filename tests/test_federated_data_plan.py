import logging
from typing import Dict, Any
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

import fedbiomed
from fedbiomed.common.constants import ProcessTypes
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError
from fedbiomed.common.training_plans import FederatedDataPlan


class MyFederatedDataPlan(FederatedDataPlan):
    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
    ) -> None:
        super().post_init(model_args, training_args)

    def training_data(self):
        pass


class TestBaseTrainingPlan(unittest.TestCase):
    def setUp(self):
        self.tp = MyFederatedDataPlan()
        logging.disable('CRITICAL')
        pass

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_federated_data_plan_01_add_dependency(self):
        """ Test  adding dependencies """

        expected = ['from torch import nn']
        self.tp.add_dependency(expected)
        self.assertListEqual(expected, self.tp._dependencies, 'Can not set dependency properly')

    def test_federated_data_plan_02_set_dataset_path(self):
        """ Test setting dataset path """

        expected = '/path/to/my/data.csv'
        self.tp.set_dataset_path(expected)
        self.assertEqual(expected, self.tp.dataset_path, 'Can not set `dataset_path` properly')

    def test_federated_data_plan_03_save_code(self):
        """ Testing the method save_code of BaseTrainingPlan """
        expected_filepath = 'path/to/model.py'

        # Test without dependencies
        with patch.object(fedbiomed.common.training_plans._federated_plan, 'open', MagicMock()) as mock_open:
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")

        # Test with adding dependencies
        with patch.object(fedbiomed.common.training_plans._federated_plan, 'open', MagicMock()) as mock_open:
            self.tp.add_dependency(['from fedbiomed.common.training_plans import TorchTrainingPlan'])
            self.tp.save_code(expected_filepath)
            mock_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")

        # Test if get_class_source raises error
        with patch('fedbiomed.common.training_plans._federated_plan.get_class_source') \
                as mock_get_class_source:
            mock_get_class_source.side_effect = FedbiomedError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

        # Test if open function raises errors
        with patch.object(fedbiomed.common.training_plans._federated_plan, 'open', MagicMock()) as mock_open:
            mock_open.side_effect = OSError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

            mock_open.side_effect = PermissionError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

            mock_open.side_effect = MemoryError
            with self.assertRaises(FedbiomedTrainingPlanError):
                self.tp.save_code(expected_filepath)

    def test_federated_data_plan_04_add_preprocess(self):
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

    def test_federated_data_plan_05_set_data_loaders(self):
        test_data_loader = [1, 2, 3]
        train_data_loader = [1, 2, 3]

        self.tp.set_data_loaders(train_data_loader, test_data_loader)
        self.assertListEqual(self.tp.training_data_loader, train_data_loader)
        self.assertListEqual(self.tp.testing_data_loader, test_data_loader)

    def test_federated_data_plan_06_infer_batch_size(self):
        """Test that the utility to infer batch size works correctly.

        Supported data types are:
            - torch tensor
            - numpy array
            - dict mapping {modality: tensor/array}
            - tuple or list containing the above
        """
        batch_size = 4
        tp = MyFederatedDataPlan()

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
