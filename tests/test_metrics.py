import unittest

from fedbiomed.common.metrics import Metrics
from fedbiomed.common.constants import MetricTypes
from fedbiomed.common.exceptions import FedbiomedMetricError

import numpy as np


class TestMetrics(unittest.TestCase):
    """
    Test the Metrics class
    """

    # before the tests
    def setUp(self):
        self.metrics = Metrics()
        pass

    # after the tests
    def tearDown(self):
        pass

    def test_metrics_01_evaluate_errors(self):
        """Testing exceptions for evaluate method of metrics"""

        # Test invalid y_pred
        with self.assertRaises(FedbiomedMetricError):
            y_true = 'toto'
            y_pred = [1, 2, 3]
            self.metrics.evaluate(y_true=y_true, y_pred=y_pred, metric=MetricTypes.ACCURACY)

        # Test invalid y_true
        with self.assertRaises(FedbiomedMetricError):
            y_true = [1, 2, 3]
            y_pred = 'toto'
            self.metrics.evaluate(y_true=y_true, y_pred=y_pred, metric=MetricTypes.ACCURACY)

        # Test invalid metric type
        with self.assertRaises(FedbiomedMetricError):
            y_true = [0, 0, 1, 0]
            y_pred = [0, 1, 1, 1]
            self.metrics.evaluate(y_true=y_true, y_pred=y_pred, metric='DDD')

    def test_metrics_02_evaluate(self):
        """Testing evaluate method of metrics"""

        # Test both are 1D array with labels
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')

        # Test both are 1D array with labels as string
        y_true = ['0', '1', '0', '1']
        y_pred = ['0', '1', '0', '1']
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')

        # Test both are 1D array with labels as string
        y_true = ['0', '1', '2', '1']
        y_pred = ['0', '1', '2', '1']
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')


        # Test y_true is 2D array and y_pred 1D array with num labels
        y_true = [[1, 0], [0, 1], [1, 0], [0, 1]]
        y_pred = [0, 1, 0, 1]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')

        # Test y_true and y_pred are 2D arrays
        y_true = [[1, 0], [0, 1], [1, 0], [0, 1]]
        y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')

        # Test y_true is 1D and y_pred is 2D array
        y_true = [0, 1, 0, 1]
        y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')

        # Binary: test y_true is 1D and y_pred is 1D array with probs
        y_true = [0, 1, 0, 1]
        y_pred = [0.2, 0.6, 0.01, 0.8]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')


        # Multiclass: test y_true is 1D and y_pred is 1D array
        # Raises error since, metric is classification metric and y_true is continues
        with self.assertRaises(FedbiomedMetricError):
            y_true = [2.5, 0.1, 1.1, 2.2]
            y_pred = [2.5, 0.1, 1.2, 2.2]
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)

        # Multiclass: test y_true is 2D and y_pred is 2D array
        y_true = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        y_pred = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not calculate Accuracy correctly')

        # Multiclass: test y_true is 2D and y_pred is 2D array as float values
        y_true = [[0.5, -2, 2], [0.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        y_pred = [[0.5, -2, 2], [0.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not calculate Accuracy correctly')

        # Multiclass: test y_true is 1D and y_pred is 2D array as float values
        y_true = [2, 0, 1, 2]
        y_pred = [[0.5, -2, 2], [2.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not calculate Accuracy correctly')

        # Multiclass: test y_true is 1D and y_pred is 1D array
        y_true = [2, 0, 1, 2]
        y_pred = [2, 0, 1, 2]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not calculate Accuracy correctly')

        # Test both are 1D array with labels as string
        y_true = ['0', '1', '2', '1']
        y_pred = ['0', '1', '2', '1']
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not calculate F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not calculate Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not calculate Precision correctly')


    def test_metrics_01_accuracy(self):
        pass

    def test_metrics_02_precision(self):
        pass

    def test_metrics_03_avg_precision(self):
        pass

    def test_metrics_04_recall(self):
        pass

    def test_metrics_05_roc_auc(self):
        pass

    def test_metrics_06_f1_score(self):
        pass

    def test_metrics_07_mse(self):
        pass

    def test_metrics_08_mae(self):
        pass

    def test_metrics_09_explained_variance(self):
        pass

    def test_metrics_10_default_metric(self):
        pass


if __name__ == '__main__':
    unittest.main()
