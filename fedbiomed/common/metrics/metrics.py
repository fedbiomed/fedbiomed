from sklearn import metrics
import torch
import numpy as np

from fedbiomed.common.constants import MetricTypes,ErrorNumbers
from fedbiomed.common.logger     import logger
from fedbiomed.common.exceptions import FedbiomedMetricError

class Metrics():
    """
    Performance metrics used in training/testing evaluation.
    This class return sklearn metrics after performing sanity check on predictions and true values inputs.
    All inputs of type tensor torch are transformed to numpy array.

    Parameters:
    ----------
    y_true: array-like or torch.Tensor
            Ground Truth (correct) target values or labels.
    y_pred: array-like or torch.Tensor
            Estimated target values or Predicted labels.
    y_score: array_like or torch.Tensor, default: None
            Target scores.

    Attributes:
    ----------
        Y_true: array-like
            Ground Truth (correct) target values or labels.
        Y_pred: array-like
            Estimated target values or Predicted labels.
        Y_score: array-like
            Target scores.

        metric: dict { MetricTypes : skleran.metrics }
            Dictionary of keys value in  MetricTypes values: { ACCURACY, F1_SCORE, PRECISION, AVG_PRECISION, RECALL, ROC_AUC, MEAN_SQUARE_ERROR, MEAN_ABSOLUTE_ERROR, EXPLAINED_VARIANCE}

    """
    def __init__(self,
                 y_true,
                 y_pred,
                 y_score=None):

        self.Y_true = self._check_array(y_true)
        self.Y_pred = self._check_array(y_pred)
        self.Y_score = self._check_array(y_score)

        self.metrics = {
                MetricTypes.ACCURACY.value : self.accuracy,
                MetricTypes.PRECISION.value: self.precision,
                MetricTypes.AVG_PRECISION.value: self.avg_precision,
                MetricTypes.RECALL.value: self.recall,
                MetricTypes.ROC_AUC.value: self.roc_auc,
                MetricTypes.F1_SCORE.value: self.f1_score,
                MetricTypes.MEAN_SQUARE_ERROR.value: self.mse,
                MetricTypes.MEAN_ABSOLUTE_ERROR.value: self.mae,
                MetricTypes.EXPLAINED_VARIANCE.value: self.explained_variance,
        }

    def accuracy(self,**kwargs):
        """
        Evaluate the accuracy score
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score]
        Args:
            - normalize (bool, default=True, optional):
              If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
        Returns:
            - sklearn.metrics.accuracy_score(Y_true, Y_pred, normalize = True,sample_weight = None)
            score (float)
        """
        try:
            return metrics.accuracy_score(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def precision(self,**kwargs):
        """
        Evaluate the precision score
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html]
        Args:
             - labels (array-like, default=None, optional)
            The set of labels to include when average != 'binary', and their order if average is None.
            - pos_label (str or int, default=1, optional)
            The class to report if average='binary' and the data is binary.
            - average ({‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’, optional)
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            precision (float, or array of float of shape (n_unique_labels,))
        """
        try:
            return metrics.precision_score(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def avg_precision(self,**kwargs):
        """
        Evaluate the average precision score from prediction scores (Y_score).
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score]
        Args:
            - average ({‘micro’, ‘samples’, ‘weighted’, ‘macro’} or None, default=’macro’, optional)
            If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - pos_label (int or str, default=1, optional)
            The label of the positive class.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
        Returns:
            - sklearn.metrics.average_precision_score(Y_true, Y_score, normalize = True, sample_weight = None)
            average precision (float)
        """
        if self.Y_score is None:
            msg = ErrorNumbers.FB607.value + " For the computation of average precision score you should provide the target scores y_score "
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
            return
        try:
            return metrics.average_precision_score(self.Y_true, self.Y_score, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def recall(self, **kwargs):
        """
        Evaluate the recall.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score]
        Args:
            - labels (array-like, default=None, optional)
            The set of labels to include when average != 'binary', and their order if average is None.
            - pos_label (str or int, default=1, optional)
            The class to report if average='binary' and the data is binary.
            - average ({‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’, optional)
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            recall (float (if average is not None) or array of float of shape (n_unique_labels,))
        """
        try:
            return metrics.recall_score(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def roc_auc(self, **kwargs):
        """
        Evaluate the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score]
        Args:
            - average ({‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’, optional)
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - max_fpr (float > 0 and <= 1, default=None, optional)
            If not None, the standardized partial AUC [2] over the range [0, max_fpr] is returned. For the multiclass case, max_fpr, should be either equal to None or 1.0.
            - multi_class({‘raise’, ‘ovr’, ‘ovo’}, default=’raise’, optional)
            Only used for multiclass targets. Determines the type of configuration to use. The default value raises an error, so either 'ovr' or 'ovo' must be passed explicitly.
            - labels (array-like of shape (n_classes,), default=None, optional)
            Only used for multiclass targets. List of labels that index the classes in y_score. If None, the numerical or lexicographical order of the labels in y_true is used.
        Returns:
            - sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
            auc (float)
        """
        if self.Y_score is None:
            msg = ErrorNumbers.FB607.value + " For the computation of roc_auc you should provide the target scores y_score "
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
            return
        try:
            return metrics.roc_auc_score(self.Y_true, self.Y_score, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def f1_score(self, **kwargs):
        """
        Evaluate the F1 score.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score]
        Args:
            - labels (array-like, default=None, optional)
            The set of labels to include when average != 'binary', and their order if average is None.
            - pos_label (str or int, default=1, optional)
            The class to report if average='binary' and the data is binary.
            - average{‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            f1_score (float or array of float, shape = [n_unique_labels])
        """
        try:
            return metrics.f1_score(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def mse(self, **kwargs):
        """
        Evaluate the mean squared error.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,), default=’uniform_average’, optional)
            Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
            - squared (bool, default=True, optional)
            If True returns MSE value, if False returns RMSE value.
        Returns:
            - sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
            score (float or ndarray of floats)
        """
        try:
            return metrics.mean_squared_error(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def mae(self, **kwargs):
        """
        Evaluate the mean absolute error.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,), default=’uniform_average’, optional)
            Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        Returns:
            - sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
            score (float or ndarray of floats)
        """
        try:
            return metrics.mean_absolute_error(self.Y_true, self.Y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def explained_variance(self, **kwargs):
        """
        Evaluate the accuracy score.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’, ‘variance_weighted’} or array-like of shape (n_outputs,), default=’uniform_average’, optional)
            Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        Returns:
            - sklearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
            score (float or ndarray of floats)
        """
        try:
            return metrics.explained_variance_score(self.Y_true, self.Y_pred, **kwargs)

        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            logger.critical(msg)
            raise FedbiomedMetricError(msg)
        return

    def evaluate(self, metric=None, **kwargs):
        """
        evaluate performance.
        Args:
            - metric (MetricTypes, or str in {ACCURACY, F1_SCORE, PRECISION, AVG_PRECISION, RECALL, ROC_AUC, MEAN_SQUARE_ERROR, MEAN_ABSOLUTE_ERROR, EXPLAINED_VARIANCE}), default = None.
            The metric used to evaluate performance. If None default Metric is used: accuracy_score for classification and mean squared error for regression.
            - kwargs: The arguments specifics to each type of metrics.
        Returns:
            - score, auc, ... (float or array of floats) depending on the metric used.
        """
        if metric is not None and type(metric)==MetricTypes and metric in MetricTypes:
            result = self.metrics[metric.value](**kwargs)
        elif type(metric) == str and metric in MetricTypes.list():
            result = self.metrics[metric](**kwargs)
        else:
            result = self._get_default_metric()
        return result

    def _convert_to_array(self,X):
        """
        Convert torch tensor to numpy array

        Args:
            X: torch tensor
        Returns:
            numpy array
        """
        return X.numpy()

    def _check_array(self, array):
        """
        Check if the input is of type torch tensor and transform it to numpy array.

        Args:
            array: array-like or torch tensor
        Returns:
            array: array-like
        """
        if array is not None:
            dtype_orig = getattr(array, "dtype", None)
            if isinstance(dtype_orig, torch.dtype):
                array = self._convert_to_array(array)
        return array

    def _get_default_metric(self):
        """
        If metric name is not specified, return default metrics: accuracy_score in case of classification and mean_absolute_error in case of regression.

        Returns:
            metric: sklearn.metrics.accuracy_score or metrics.mean_squared_error
        """
        if np.array(self.Y_true).dtype == 'float':
            return metrics.accuracy_score(self.Y_true,self.Y_pred)
        else:
            return metrics.mean_squared_error(self.Y_true,self.Y_pred)

# def evaluate_metric(y_true,y_pred,metric_name=None,*args,**kwargs):
#     """
#     [TO DO] This funcion will be moved to the test/evaluation logic.
#
#     Create an instance of Metrics and evaluate the performance based on the metric name and arguments passed as input.
#
#     Parameters:
#     ----------
#     y_true: array-like or torch.Tensor
#             Ground Truth (correct) target values or labels
#     y_pred: array-like or torch.Tensor
#             Estimated target values or Predicted labels.
#     metric_name: str, default: None
#             The performance metric used for evaluation. If None mean_absolute_error is used for regression and accuracy_score for classification.
#     Returns:
#     -------
#     Performance: The output of sklearn.metrics used for evaluation.
#     """
#     evaluation = Metrics(y_true,y_pred)
#
#     acc = evaluation.evaluate(MetricTypes.ACCURACY)
#     f1 = evaluation.evaluate(MetricTypes.F1)





