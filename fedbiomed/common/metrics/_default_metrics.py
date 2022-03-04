from sklearn import metrics
import torch

class Evaluation_Metric():
    def __init__(self,X,Y,metric_name=None,*args,**kwargs):
        self.X = self._check_array(X)
        self.Y = self._check_array(Y)
        self.metric = self.evaluation_metric(metric_name)

    def _convert_to_array(self,X):
        X =  X.numpy()

    def _check_array(self, array):
        dtype_orig = getattr(array, "dtype", None)
        if isinstance(dtype_orig, torch.dtype):
            self._convert_to_array(array)
        return array

    def evaluation_metric(self,metric_name):
        if metric_name:
            metric = eval('metrics.{}'.format(metric_name))
        return metric



