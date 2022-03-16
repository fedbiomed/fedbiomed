from sklearn import metrics
import torch
import numpy as np

from fedbiomed.common.constants import MetricTypes, MetricForms, ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedMetricError


class Metrics(object):

    def __init__(self,
                 y_true: np.ndarray = None,
                 y_pred: np.ndarray = None):

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

        # At the beginning y_true and y_pred can be none and can be set by the setters
        self.Y_true = None
        self.Y_pred = None

        self.metrics = {
            MetricTypes.ACCURACY.name: self.accuracy,
            MetricTypes.PRECISION.name: self.precision,
            MetricTypes.AVG_PRECISION.name: self.avg_precision,
            MetricTypes.RECALL.name: self.recall,
            MetricTypes.ROC_AUC.name: self.roc_auc,
            MetricTypes.F1_SCORE.name: self.f1_score,
            MetricTypes.MEAN_SQUARE_ERROR.name: self.mse,
            MetricTypes.MEAN_ABSOLUTE_ERROR.name: self.mae,
            MetricTypes.EXPLAINED_VARIANCE.name: self.explained_variance,
        }

    def set_y_true(self, y_true: np.ndarray):
        """
        Setter for true values (y_true)

        Args:
            y_true:

        Raises:
            - FedbiomedMetricError
        """

        if not isinstance(y_true, np.ndarray):
            raise FedbiomedMetricError()

        self.Y_true = y_true

    def set_y_pred(self, y_pred: np.ndarray):
        """
        Setter for predicted values (y_pred)

        Args:
            y_true:

        Raises:
            - FedbiomedMetricError
        """

        if not isinstance(y_pred, np.ndarray):
            raise FedbiomedMetricError()

        self.Y_true = y_pred

    @staticmethod
    def accuracy(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 **kwargs):
        """
        Evaluate the accuracy score
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score]
        Args:
            - normalize (bool, default=True, optional):
              If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
        Returns:
            - sklearn.metrics.accuracy_score(y_true, y_pred, normalize = True,sample_weight = None)
            score (float)
        """

        try:
            return metrics.accuracy_score(y_true, y_pred, **kwargs)
        except Exception as e:
            print(e)
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def precision(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  **kwargs):
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
            return metrics.precision_score(y_true, y_pred, **kwargs)
        except Exception as e:
            print(e)
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)
        return

    @staticmethod
    def avg_precision(y_true: np.ndarray,
                      y_score: np.ndarray,
                      **kwargs):
        """
        Evaluate the average precision score from prediction scores (y_score).
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score]
        Args:
            - average ({‘micro’, ‘samples’, ‘weighted’, ‘macro’} or None, default=’macro’, optional)
            If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
            - pos_label (int or str, default=1, optional)
            The label of the positive class.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
        Returns:
            - sklearn.metrics.average_precision_score(y_true, y_score, normalize = True, sample_weight = None)
            average precision (float)
        """
        try:
            return metrics.average_precision_score(y_true, y_score, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def recall(y_true: np.ndarray,
               y_pred: np.ndarray,
               **kwargs):
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
            return metrics.recall_score(y_true, y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def roc_auc(y_true: np.ndarray,
                y_score: np.ndarray,
                **kwargs):
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

        try:
            return metrics.roc_auc_score(y_true, y_score, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def f1_score(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 **kwargs):
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

        # Check target variable is multi class or binary
        if len(np.unique(y_true)) > 2:
            average = kwargs.get('average', 'weighted')
            logger.info('Using weighted calculation for multiclass F1 SCORE')
        else:
            average = kwargs.get('average', 'binary')

        # Remove `average` parameter from **kwargs
        kwargs.pop("average", None)

        try:
            return metrics.f1_score(y_true, y_pred, average=average, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def mse(y_true: np.ndarray,
            y_pred: np.ndarray,
            **kwargs):
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
            return metrics.mean_squared_error(y_true, y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def mae(y_true: np.ndarray,
            y_pred: np.ndarray,
            **kwargs):
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
            return metrics.mean_absolute_error(y_true, y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB607.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def explained_variance(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           **kwargs):
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
            return metrics.explained_variance_score(y_true, y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    def evaluate(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 metric: MetricTypes,
                 **kwargs):
        """
        evaluate performance.
        Args:
            - metric (MetricTypes, or str in {ACCURACY, F1_SCORE, PRECISION, AVG_PRECISION, RECALL, ROC_AUC, MEAN_SQUARE_ERROR, MEAN_ABSOLUTE_ERROR, EXPLAINED_VARIANCE}), default = None.
            The metric used to evaluate performance. If None default Metric is used: accuracy_score for classification and mean squared error for regression.
            - kwargs: The arguments specifics to each type of metrics.
        Returns:
            - score, auc, ... (float or array of floats) depending on the metric used.
        """
        if not isinstance(metric, MetricTypes):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Metric should instance of `MetricTypes`")

        if y_true is not None and not isinstance(y_true, np.ndarray):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_true` should an instance "
                                       f"of `np.ndarray`, but got {type(y_true)} ")

        if y_pred is not None and not isinstance(y_pred, np.ndarray):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_pred` should an instance "
                                       f"of `np.ndarray`, but got {type(y_true)} ")

        # if set(y_true.shape) != set(y_pred.shape):
        #     raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Shape of true `y_true` and predicted `y_pred` "
        #                                f"does not match; {y_true.shape}, {y_pred.shape}")

        y_pred = self._configure_y_pred_based_on_metric_form(y_pred=y_pred, metric=metric)
        result = self.metrics[metric.name](y_true, y_pred, **kwargs)

        return result

    @staticmethod
    def _configure_y_pred_based_on_metric_form(y_pred: np.ndarray,
                                               metric: MetricTypes):
        """

        """
        # Get shape of the prediction should be 1D or 2D array
        y_pred = np.squeeze(y_pred)
        shape = y_pred.shape

        # Shape of the prediction array should be (samples, outputs) or (samples, )
        if len(shape) > 2:
            raise FedbiomedMetricError()

        output_shape = shape[1] if len(shape) == 2 else 0  # 0 for 1D array

        if metric.value[1] is MetricForms.CLASSIFICATION_LABELS:
            if output_shape == 0:
                # TODO: Get threshold value from researcher
                y_pred = np.where(y_pred > 0.5, 1, 0)
            else:
                y_pred = np.argmax(y_pred, axis=1)

        elif metric.value[1] is MetricForms.REGRESSION:
            if output_shape > 0:
                raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: For the metric `{metric.name}` multiple "
                                           f"output regression is not supported")

        return y_pred

    # def _convert_to_array(self, X):
    #     """
    #     Convert torch tensor to numpy array
    #
    #     Args:
    #         X: torch tensor
    #     Returns:
    #         numpy array
    #     """
    #     return X.numpy()
    #
    # def _check_array(self, array):
    #     """
    #     Check if the input is of type torch tensor and transform it to numpy array.
    #
    #     Args:
    #         array: array-like or torch tensor
    #     Returns:
    #         array: array-like
    #     """
    #     if array is not None:
    #         dtype_orig = getattr(array, "dtype", None)
    #         if isinstance(dtype_orig, torch.dtype):
    #             array = self._convert_to_array(array)
    #     return array
    #
    # def _get_default_metric(self):
    #     """
    #     If metric name is not specified, return default metrics: accuracy_score in case of classification and mean_absolute_error in case of regression.
    #
    #     Returns:
    #         metric: sklearn.metrics.accuracy_score or metrics.mean_squared_error
    #     """
    #     if np.array(self.Y_true).dtype == 'float':
    #         return metrics.accuracy_score(self.Y_true, self.Y_pred)
    #     else:
    #         return metrics.mean_squared_error(self.Y_true, self.Y_pred)
