import unittest
from unittest.mock import patch

from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.exceptions import FedbiomedMetricError


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

    def test_metrics_01_evaluate_base_errors(self):
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

    def test_metrics_02_evaluate_binary_classification_1D_array(self):
        """Testing evaluate method of metrics"""

        # Test both are 1D array with labels
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not calculate Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_03_evaluate_binary_classification_1D_aray_string(self):
        # Test both are 1D array with labels as string
        y_true = ['0', '1', '0', '1']
        y_pred = ['0', '1', '0', '1']
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_04_evaluate_binary_classification_2D_1D_array(self):
        # Test y_true is 2D array and y_pred 1D array with num labels
        y_true = [[1, 0], [0, 1], [1, 0], [0, 1]]
        y_pred = [0, 1, 0, 1]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_05_evaluate_binary_classification_2D_2D_array(self):
        # Test y_true and y_pred are 2D arrays
        y_true = [[1, 0], [0, 1], [1, 0], [0, 1]]
        y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_06_evaluate_binary_classification_2D_2D_array(self):
        # Test y_true is 1D and y_pred is 2D array
        y_true = [0, 1, 0, 1]
        y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_07_evaluate_binary_classification_2D_1D_array_with_probs(self):
        # Binary: test y_true is 1D and y_pred is 1D array with probs
        y_true = [0, 1, 0, 1]
        y_pred = [0.2, 0.6, 0.01, 0.8]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Binary: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Binary: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Binary: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Binary: Could not compute Precision correctly')

    def test_metrics_08_evaluate_multiclass_classification_1D_1D_array_errors(self):
        """ Multiclass: test y_true is 1D and y_pred is 1D array """

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

    def test_metrics_09_evaluate_multiclass_classification_2D_2D_array(self):
        """Multiclass: test y_true is 2D and y_pred is 2D array"""
        y_true = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        y_pred = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Multiclass: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Precision correctly')

    def test_metrics_10_evaluate_multiclass_classification_2D_2D_array_probs(self):
        """Multiclass: test y_true is 2D and y_pred is 2D array as float values"""

        y_true = [[0.5, -2, 2], [0.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        y_pred = [[0.5, -2, 2], [0.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Multiclass: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Precision correctly')

    def test_metrics_11_evaluate_multiclass_classification_1D_2D_array_probs(self):
        """ Multiclass: test y_true is 1D and y_pred is 2D array as float values"""

        y_true = [2, 0, 1, 2]
        y_pred = [[0.5, -2, 2], [2.1, 1.5, 0.1], [-1.5, 1.2, 0.4], [-2.5, 1, 2.6]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Multiclass: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Precision correctly')

    def test_metrics_12_evaluate_multiclass_classification_1D_1D_array(self):
        """Multiclass: test y_true is 1D and y_pred is 1D array"""

        y_true = [2, 0, 1, 2]
        y_pred = [2, 0, 1, 2]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Multiclass: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Precision correctly')

    def test_metrics_13_evaluate_multiclass_classification_1D_1D_array_strings(self):
        """ Multiclass: Test both are 1D array with labels as string """

        y_true = ['0', '1', '2', '1']
        y_pred = ['0', '1', '2', '1']
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Accuracy correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.F1_SCORE)
        self.assertEqual(result, 1, 'Multiclass: Could not compute F1 Score correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.RECALL)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Recall correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.PRECISION)
        self.assertEqual(result, 1, 'Multiclass: Could not compute Precision correctly')

    def test_metrics_14_evaluate_regression_1D_1D_array_strings(self):
        """ Multiclass: Test both are 1D array with labels as string """

        # Test exception if y_true and y_pred is in string type and metric is one of regression metrics
        y_true = ['0', '1', '2', '1']
        y_pred = ['0', '1', '2', '1']
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_SQUARE_ERROR)

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_ABSOLUTE_ERROR)

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.EXPLAINED_VARIANCE)

        y_true = [12, 13, 14, 15]
        y_pred = [11, 12, 13, 14]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_SQUARE_ERROR)
        self.assertEqual(result, 1, 'Could not compute MEAN_SQUARE_ERROR correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_ABSOLUTE_ERROR)
        self.assertEqual(result, 1, 'Could not compute MEAN_ABSOLUTE_ERROR correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.EXPLAINED_VARIANCE)
        self.assertEqual(result, 1, 'Could not compute EXPLAINED_VARIANCE correctly')

        # Should also calculate classification based metrics
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)
        self.assertEqual(result, 0, 'Could not compute ACCURACY for regression input correctly')

        # Test multi output
        y_true = [[12, 12], [13, 13], [14, 14], [15, 15]]
        y_pred = [[11, 11], [12, 12], [13, 13], [14, 14]]
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_SQUARE_ERROR)
        self.assertListEqual(list(result), [1, 1], 'Could not compute MEAN_SQUARE_ERROR correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_ABSOLUTE_ERROR)
        self.assertListEqual(list(result), [1, 1], 'Could not compute MEAN_ABSOLUTE_ERROR correctly')
        result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.EXPLAINED_VARIANCE)
        self.assertListEqual(list(result), [1, 1], 'Could not compute EXPLAINED_VARIANCE correctly')

        # Test missmatch shape for regression metrics should raise exception
        y_true = [[12, 12], [13, 13], [14, 14], [15, 15]]
        y_pred = [11, 12, 13 ,14]
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.MEAN_SQUARE_ERROR)

    def test_metrics_15_evaluate_shape_errors(self):
        """ Multiclass: Testing error due to y_true and y_pred shapes """

        y_true = [[0, 1], [0, 1], [0, 1], [0, 1]]
        y_pred = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        with self.assertRaises(FedbiomedMetricError):
            result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)

        y_true = [[0, 1], [0, 1], ]
        y_pred = [[[0, 1], [0,1]], [[0, 1], [0, 1]]]
        with self.assertRaises(FedbiomedMetricError):
            result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)

        y_true = [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
        y_pred = [[0, 1], [0, 1]]
        with self.assertRaises(FedbiomedMetricError):
            result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)

        y_true = [[0, 1], [0, 1], [0, 1], [0, 1]]
        y_pred = [[0, 1], [0, 1]]
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)

    def test_metrics_16_evaluate_multiclass_classification_1D_1D_array_strings_errors(self):
        """Test if predicted values and true values are not of same type"""

        y_true = ['0', '1', '2', '1']
        y_pred = [0, 1, 2, 1]
        with self.assertRaises(FedbiomedMetricError):
            result = self.metrics.evaluate(y_true, y_pred, metric=MetricTypes.ACCURACY)

    @patch('fedbiomed.common.metrics.metrics.accuracy_score')
    @patch('fedbiomed.common.metrics.metrics.precision_score')
    @patch('fedbiomed.common.metrics.metrics.recall_score')
    @patch('fedbiomed.common.metrics.metrics.f1_score')
    @patch('fedbiomed.common.metrics.metrics.mean_squared_error')
    @patch('fedbiomed.common.metrics.metrics.mean_absolute_error')
    @patch('fedbiomed.common.metrics.metrics.explained_variance_score')
    def test_metrics_17_try_expect_blocks_of_eval_functions(self,
                                                            patch_exp_variance,
                                                            patch_mean_abs,
                                                            patch_mean_sq,
                                                            patch_f1_score,
                                                            patch_recall_score,
                                                            patch_precision_score,
                                                            patch_accuracy):
        patch_accuracy.side_effect = Exception
        patch_precision_score.side_effect = Exception
        patch_recall_score.side_effect = Exception
        patch_f1_score.side_effect = Exception
        patch_mean_sq.side_effect = Exception
        patch_mean_abs.side_effect = Exception
        patch_exp_variance.side_effect = Exception

        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.accuracy(y_true, y_pred)

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.f1_score(y_true, y_pred)

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.recall(y_true, y_pred)

        with self.assertRaises(FedbiomedMetricError):
            self.metrics.precision(y_true, y_pred)

        y_true = [12, 13, 14, 15]
        y_pred = [11, 12, 13, 14]
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.explained_variance(y_true, y_pred)
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.mae(y_true, y_pred)
        with self.assertRaises(FedbiomedMetricError):
            self.metrics.mse(y_true, y_pred)


if __name__ == '__main__':
    unittest.main()
