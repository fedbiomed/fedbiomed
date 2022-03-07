from sklearn import metrics
import torch
import numpy as np

class Metrics():
    """
    Performance metrics used in training/testing evaluation.
    This class return sklearn metrics after performing sanity check on predictions and true values inputs.
    All inputs of type tensor torch are transformed to numpy array.

    Parameters:
    ----------
    y_true: array-like or torch.Tensor
            Ground Truth (correct) target values or labels
    y_pred: array-like or torch.Tensor
            Estimated target values or Predicted labels.
    metric_name: str, default: None
            The performance metric used for evaluation. If None mean_absolute_error is used for regression and accuracy_score for classification.

    Attributes:
    ----------
        Y_true: array-like
            Ground Truth (correct) target values or labels
        Y_pred: array-like
            Estimated target values or Predicted labels.
        metric: sklearn.metric
            The performance metric used for evaluation.
    """
    def __init__(self,y_true,y_pred,metric_name=None):
        self.Y_true = self._check_array(y_true)
        self.Y_pred = self._check_array(y_pred)
        self.metric = self._evaluation_metric(metric_name)

    def _convert_to_array(self,X):
        """
        Convert torch tensor to numpy array

        Parameters:
        ----------
            X: torch tensor
        Returns:
        -------
            None
        """
        X =  X.numpy()

    def _check_array(self, array):
        """
        Check if the input is of type torch tensor and transform it to numpy array.

        Parameters:
        ----------
            array: array-like or torch tensor
        Returns:
        -------
            array: array-like
        """
        dtype_orig = getattr(array, "dtype", None)
        if isinstance(dtype_orig, torch.dtype):
            self._convert_to_array(array)
        return array

    def _evaluation_metric(self,metric_name=None):
        """
        select the sklearn evaluation metric.

        Parameters:
        ----------
            metric_name: str, default: None
        Returns:
        -------
            metric: sklearn metric
        """
        if metric_name:
            metric = eval('metrics.{}'.format(metric_name))
        else:
            metric = self._get_default_metric()
        return metric

    def _get_default_metric(self):
        """
        If metric name is not specified, return default metrics: accuracy_score in case of classification and mean_absolute_erro in case of regression.

        Returns:
        -------
            metric: sklearn.metrics
        """
        if np.array(self.Y_true).dtype == 'float':
            metric = metrics.mean_absolute_error
        else:
            metric = metrics.accuracy_score
        return metric

def evaluate_metric(y_true,y_pred,metric_name=None,*args,**kwargs):
    """
    Create an instance of Metrics and evaluate the performance based on the metric name and arguments passed as input.

    Parameters:
    ----------
    y_true: array-like or torch.Tensor
            Ground Truth (correct) target values or labels
    y_pred: array-like or torch.Tensor
            Estimated target values or Predicted labels.
    metric_name: str, default: None
            The performance metric used for evaluation. If None mean_absolute_error is used for regression and accuracy_score for classification.

    Returns:
    -------
    Performance: The output of sklearn.metrics used for evaluation.
    """
    evaluationMetric = Metrics(y_true,y_pred,metric_name)
    return evaluationMetric.metric(y_true,y_pred,*args,**kwargs)




