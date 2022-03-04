from sklearn import metrics
import torch
import numpy as np

class Metrics():
    def __init__(self,y_true,y_pred,metric_name=None):
        self.Y_true = self._check_array(y_true)
        self.Y_pred = self._check_array(y_pred)
        self.metric = self._evaluation_metric(metric_name)

    def _convert_to_array(self,X):
        X =  X.numpy()

    def _check_array(self, array):
        dtype_orig = getattr(array, "dtype", None)
        if isinstance(dtype_orig, torch.dtype):
            self._convert_to_array(array)
        return array

    def _evaluation_metric(self,metric_name=None):
        if metric_name:
            metric = eval('metrics.{}'.format(metric_name))
        else:
            metric = self._get_default_metric()
        return metric

    def _get_default_metric(self):
        if np.array(self.Y_true).dtype == 'float':
            metric = metrics.mean_absolute_error
        else:
            metric = metrics.accuracy_score
        return metric

def evaluate_metric(y_true,y_pred,metric_name=None,*args,**kwargs):
    evaluationMetric = Metrics(y_true,y_pred,metric_name)
    return evaluationMetric.metric(y_true,y_pred,*args,**kwargs)




