import os
import sys
import unittest
import fedbiomed.common.training_plans._base_training_plan

from unittest.mock import patch, MagicMock
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError

from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan


# Dummy Class for testing its source --------------------
class TestClass:
    def __init__(self):
        pass


# -------------------------------------------------------


class TestBaseTrainingPlan(unittest.TestCase):
    """ Test Class for Base Training Plan """

    def setUp(self):
        self.tp = BaseTrainingPlan()
        pass

    def tearDown(self) -> None:
        pass

    def test_base_training_plan_01_add_dependency(self):
        """ Test  adding dependencies """

        expected = ['from torch import nn']
        self.tp.add_dependency(expected)
        self.assertListEqual(expected, self.tp.dependencies, 'Can not set dependency properly')

    def test_base_training_plan_02_set_dataset_path(self):
        """ Test setting dataset path """

        expected = '/path/to/my/data.csv'
        self.tp.set_dataset_path(expected)
        self.assertEqual(expected, self.tp.dataset_path, 'Can not set `dataset_path` properly')

    def test_base_training_plan_03_save_code(self):
        """ Testing the method save_code of BaseTrainingPlan """

        # Test without dependencies
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
            mock_open.write.return_value = None
            mock_open.close.return_value = None

            expected_filepath = 'path/to/model.py'
            path, _ = self.tp.save_code(expected_filepath)
            self.assertEqual(path, expected_filepath, 'Can not save model file properly')

        # Test with adding dependndicies
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
            mock_open.write.return_value = None
            mock_open.close.return_value = None

            self.tp.add_dependency(['from fedbiomed.common.training_plans import TorchTrainingPlan'])
            expected_filepath = 'path/to/model.py'
            path, _ = self.tp.save_code(expected_filepath)
            self.assertEqual(path, expected_filepath, 'Can not save model file properly')

        # Test if get_class_source raises error
        with patch('fedbiomed.common.training_plans._base_training_plan.get_class_source') \
                as mock_get_class_source:
            mock_get_class_source.side_effect = FedbiomedError
            with self.assertRaises(FedbiomedTrainingPlanError):
                path, _ = self.tp.save_code(expected_filepath)

        # Test if open function raises errors
        with patch.object(fedbiomed.common.training_plans._base_training_plan, 'open', MagicMock()) as mock_open:
            mock_open.write.return_value = None
            mock_open.close.return_value = None

            mock_open.side_effect = OSError
            with self.assertRaises(FedbiomedTrainingPlanError):
                path, _ = self.tp.save_code(expected_filepath)

            mock_open.side_effect = PermissionError
            with self.assertRaises(FedbiomedTrainingPlanError):
                path, _ = self.tp.save_code(expected_filepath)

            mock_open.side_effect = MemoryError
            with self.assertRaises(FedbiomedTrainingPlanError):
                path, _ = self.tp.save_code(expected_filepath)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
