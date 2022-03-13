import unittest

from fedbiomed.common.metrics import Metrics
from fedbiomed.common.constants import MetricTypes
from fedbiomed.common.exceptions import FedbiomedMetricError

from sklearn import metrics as sk_metrics

import torch
import numpy as np

class TestMetrics(unittest.TestCase):
    """
    Test the Metrics class
    """

    # before the tests
    def setUp(self):
        self.y_true_cls_np = np.array([0, 0, 1, 1])
        self.y_true_cls_torch = torch.tensor([0,0,1,1])

        self.y_pred_cls_np = np.array([1,0,1,0])
        self.y_pred_cls_torch = torch.tensor([1,0,1,0])

        self.y_scores_cls_np =  np.array([0.1, 0.4, 0.35, 0.8],dtype='float32')
        self.y_scores_cls_torch = torch.tensor([0.1, 0.4, 0.35, 0.8],dtype=torch.float32)

        self.y_true_reg_np = np.array([3, -0.5, 2, 7],dtype='float32')
        self.y_true_reg_torch = torch.tensor([3, -0.5, 2, 7],dtype=torch.float32)

        self.y_pred_reg_np = np.array([2.5, 0.0, 2, 8],dtype='float32')
        self.y_pred_reg_torch = torch.tensor([2.5, 0.0, 2, 8],dtype=torch.float32)

        pass

    # after the tests
    def tearDown(self):
        pass

    def test_metrics_01_accuracy(self):
        sk_acc = sk_metrics.accuracy_score(self.y_true_cls_np,self.y_pred_cls_np)

        evaluation = Metrics(y_true = self.y_true_cls_np,y_pred= self.y_pred_cls_np)
        acc = evaluation.evaluate(MetricTypes.ACCURACY)
        self.assertEqual(acc, sk_acc)

        evaluation = Metrics(y_true = self.y_true_cls_torch,y_pred= self.y_pred_cls_torch)
        acc = evaluation.evaluate(MetricTypes.ACCURACY)
        self.assertEqual(acc, sk_acc)

        acc = evaluation.evaluate('ACCURACY')
        self.assertEqual(acc, sk_acc)

    def test_metrics_02_precision(self):
        sk_precision = sk_metrics.precision_score(self.y_true_cls_np, self.y_pred_cls_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np)
        precision = evaluation.evaluate(MetricTypes.PRECISION)
        self.assertEqual(precision, sk_precision)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch)
        precision = evaluation.evaluate(MetricTypes.PRECISION)
        self.assertEqual(precision, sk_precision)

    def test_metrics_03_avg_precision(self):
        sk_avg_precision = sk_metrics.average_precision_score(self.y_true_cls_np, self.y_scores_cls_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np,y_score=self.y_scores_cls_np)
        avg_precision = evaluation.evaluate(MetricTypes.AVG_PRECISION)
        self.assertEqual(avg_precision, sk_avg_precision)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch, y_score=self.y_scores_cls_torch)
        avg_precision = evaluation.evaluate(MetricTypes.AVG_PRECISION)
        self.assertEqual(avg_precision, sk_avg_precision)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch)
        with self.assertRaises(FedbiomedMetricError):
            evaluation.evaluate(MetricTypes.AVG_PRECISION)

    def test_metrics_04_recall(self):
        sk_recall = sk_metrics.recall_score(self.y_true_cls_np, self.y_pred_cls_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np)
        recall = evaluation.evaluate(MetricTypes.RECALL)
        self.assertEqual(recall, sk_recall)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch)
        recall = evaluation.evaluate(MetricTypes.RECALL)
        self.assertEqual(recall, sk_recall)

    def test_metrics_05_roc_auc(self):
        sk_roc_auc = sk_metrics.roc_auc_score(self.y_true_cls_np, self.y_scores_cls_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np, y_score=self.y_scores_cls_np)
        roc_auc = evaluation.evaluate(MetricTypes.ROC_AUC)
        self.assertEqual(roc_auc, sk_roc_auc)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch,
                             y_score=self.y_scores_cls_torch)
        roc_auc = evaluation.evaluate(MetricTypes.ROC_AUC)
        self.assertEqual(roc_auc, sk_roc_auc)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch)
        with self.assertRaises(FedbiomedMetricError):
            evaluation.evaluate(MetricTypes.ROC_AUC)

    def test_metrics_06_f1_score(self):
        sk_f1_score = sk_metrics.f1_score(self.y_true_cls_np, self.y_pred_cls_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np)
        f1_score = evaluation.evaluate(MetricTypes.F1_SCORE)
        self.assertEqual(f1_score, sk_f1_score)

        evaluation = Metrics(y_true=self.y_true_cls_torch, y_pred=self.y_pred_cls_torch)
        f1_score = evaluation.evaluate(MetricTypes.F1_SCORE)
        self.assertEqual(f1_score, sk_f1_score)

    def test_metrics_07_mse(self):
        sk_mse = sk_metrics.mean_squared_error(self.y_true_reg_np, self.y_pred_reg_np)

        evaluation = Metrics(y_true=self.y_true_reg_np, y_pred=self.y_pred_reg_np)
        mse = evaluation.evaluate(MetricTypes.MEAN_SQUARE_ERROR)
        self.assertEqual(mse, sk_mse)

        evaluation = Metrics(y_true=self.y_true_reg_torch, y_pred=self.y_pred_reg_torch)
        mse = evaluation.evaluate(MetricTypes.MEAN_SQUARE_ERROR)
        self.assertEqual(mse, sk_mse)

    def test_metrics_08_mae(self):
        sk_mae = sk_metrics.mean_absolute_error(self.y_true_reg_np, self.y_pred_reg_np)

        evaluation = Metrics(y_true=self.y_true_reg_np, y_pred=self.y_pred_reg_np)
        mae = evaluation.evaluate(MetricTypes.MEAN_ABSOLUTE_ERROR)
        self.assertEqual(mae, sk_mae)

        evaluation = Metrics(y_true=self.y_true_reg_torch, y_pred=self.y_pred_reg_torch)
        mae = evaluation.evaluate(MetricTypes.MEAN_ABSOLUTE_ERROR)
        self.assertEqual(mae, sk_mae)

    def test_metrics_09_explained_variance(self):
        sk_ev = sk_metrics.explained_variance_score(self.y_true_reg_np, self.y_pred_reg_np)

        evaluation = Metrics(y_true=self.y_true_reg_np, y_pred=self.y_pred_reg_np)
        ev = evaluation.evaluate(MetricTypes.EXPLAINED_VARIANCE)
        self.assertEqual(ev, sk_ev)

        evaluation = Metrics(y_true=self.y_true_reg_torch, y_pred=self.y_pred_reg_torch)
        ev = evaluation.evaluate(MetricTypes.EXPLAINED_VARIANCE)
        self.assertEqual(ev, sk_ev)

    def test_metrics_10_default_metric(self):
        sk_acc = sk_metrics.accuracy_score(self.y_true_cls_np, self.y_pred_cls_np)
        sk_mse = sk_metrics.mean_squared_error(self.y_true_reg_np, self.y_pred_reg_np)

        evaluation = Metrics(y_true=self.y_true_cls_np, y_pred=self.y_pred_cls_np)
        result = evaluation.evaluate()
        self.assertEqual(result, sk_acc)

        evaluation = Metrics(y_true=self.y_true_reg_np, y_pred=self.y_pred_reg_np)
        result = evaluation.evaluate("incorrect_value")
        self.assertEqual(result, sk_mse)


if __name__ == '__main__':
    unittest.main()
