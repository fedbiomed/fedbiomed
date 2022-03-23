import numpy as np

from typing import Union
from sklearn import metrics
from copy import copy
from fedbiomed.common.constants import _BaseEnum, ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedMetricError


class _MetricCategory(_BaseEnum):

    CLASSIFICATION_LABELS = 0  # return labels
    CLASSIFICATION_SCORES = 1  # return proba
    REGRESSION = 2


class MetricTypes(_BaseEnum):
    """
    List of Performance metrics used to evaluate the model.
    """

    ACCURACY = (0, _MetricCategory.CLASSIFICATION_LABELS)
    F1_SCORE = (1, _MetricCategory.CLASSIFICATION_LABELS)
    PRECISION = (2, _MetricCategory.CLASSIFICATION_LABELS)
    AVG_PRECISION = (3, _MetricCategory.CLASSIFICATION_SCORES)
    RECALL = (4, _MetricCategory.CLASSIFICATION_LABELS)
    ROC_AUC = (5, _MetricCategory.CLASSIFICATION_SCORES)

    MEAN_SQUARE_ERROR = (6, _MetricCategory.REGRESSION)
    MEAN_ABSOLUTE_ERROR = (7, _MetricCategory.REGRESSION)
    EXPLAINED_VARIANCE = (8, _MetricCategory.REGRESSION)

    def __init__(self, idx: int, metric_category: _MetricCategory) -> None:
        self._idx = idx
        self._metric_category = metric_category

    def metric_category(self) -> _MetricCategory:
        return self._metric_cetegory

class Metrics(object):

    def __init__(self):

        """
        Class of performance metrics used in testing evaluation.

        Attrs:
            metrics: dict { MetricTypes : skleran.metrics }
                Dictionary of keys values in  MetricTypes values: { ACCURACY, F1_SCORE, PRECISION,
                RECALL, ROC_AUC, MEAN_SQUARE_ERROR, MEAN_ABSOLUTE_ERROR, EXPLAINED_VARIANCE}

        """

        self.metrics = {
            MetricTypes.ACCURACY.name: self.accuracy,
            MetricTypes.PRECISION.name: self.precision,
            MetricTypes.RECALL.name: self.recall,
            MetricTypes.F1_SCORE.name: self.f1_score,
            MetricTypes.MEAN_SQUARE_ERROR.name: self.mse,
            MetricTypes.MEAN_ABSOLUTE_ERROR.name: self.mae,
            MetricTypes.EXPLAINED_VARIANCE.name: self.explained_variance,
        }

    def evaluate(self,
                 y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list],
                 metric: MetricTypes,
                 **kwargs):
        """
        Evaluate method to perform evaluation based on given metric. This method configures given y_pred
        and y_true to make them compatible with default evaluation methods.

        Args:
            - y_true (np.ndarray): True values
            - y_pred (np.ndarray): Predicted values
            - metric (MetricTypes): An instance of MetricTypes to chose metric that will be used for evaluation
            - kwargs: The arguments specifics to each type of metrics.
        Returns:
            - int or float as result of the evaluation metric

        Raises:
            - FedbiomedMetricError: in case of invalid metric, y_true and y_pred types
        """

        if not isinstance(metric, MetricTypes):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Metric should instance of `MetricTypes`")

        if y_true is not None and not isinstance(y_true, (np.ndarray, list)):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_true` should an instance "
                                       f"of `np.ndarray`, but got {type(y_true)} ")

        if y_pred is not None and not isinstance(y_pred, (np.ndarray, list)):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_pred` should an instance "
                                       f"of `np.ndarray`, but got {type(y_true)} ")

        y_true, y_pred = self._configure_y_true_pred_(y_true=y_true, y_pred=y_pred, metric=metric)

        result = self.metrics[metric.name](y_true, y_pred, **kwargs)

        return result

    @staticmethod
    def accuracy(y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list],
                 **kwargs):
        """
        Evaluate the accuracy score
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html]
        Args:
            - normalize (bool, default=True, optional):
              If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly
              classified samples.
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
    def precision(y_true: Union[np.ndarray, list],
                  y_pred: Union[np.ndarray, list],
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
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
            returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            precision (float, or array of float of shape (n_unique_labels,))
        """

        # Get average and pob_label argument based on multiclass status
        average, pos_label = Metrics._configure_multiclass_parameters(y_true, kwargs, 'PRECISION')

        kwargs.pop("average", None)
        kwargs.pop("pos_label", None)

        try:
            return metrics.precision_score(y_true, y_pred, average=average, pos_label=pos_label, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `PRECISION` "
                                       f"calculation: {str(e)}")

    @staticmethod
    def recall(y_true: Union[np.ndarray, list],
               y_pred: Union[np.ndarray, list],
               **kwargs):
        """
        Evaluate the recall.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html]
        Args:
            - labels (array-like, default=None, optional)
            The set of labels to include when average != 'binary', and their order if average is None.
            - pos_label (str or int, default=1, optional)
            The class to report if average='binary' and the data is binary.
            - average ({‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’, optional)
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
            returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
            sample_weight=None, zero_division='warn')
            recall (float (if average is not None) or array of float of shape (n_unique_labels,))
        """

        # Get average and pob_label argument based on multiclass status
        average, pos_label = Metrics._configure_multiclass_parameters(y_true, kwargs, 'RECALL')

        kwargs.pop("average", None)
        kwargs.pop("pos_label", None)

        try:
            return metrics.recall_score(y_true, y_pred, average=average, pos_label=pos_label, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `RECALL` "
                                       f"calculation: {str(e)}")

    @staticmethod
    def f1_score(y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list],
                 **kwargs):
        """
        Evaluate the F1 score.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html]
        Args:
            - labels (array-like, default=None, optional)
            The set of labels to include when average != 'binary', and their order if average is None.
            - pos_label (str or int, default=1, optional)
            The class to report if average='binary' and the data is binary.
            - average{‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
            returned. Otherwise, this determines the type of averaging performed on the data.
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - zero_division (“warn”, 0 or 1, default=”warn”, optional)
            Sets the value to return when there is a zero division.
        Returns:
            - sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
            sample_weight=None, zero_division='warn')
            f1_score (float or array of float, shape = [n_unique_labels])
        """

        # Get average and pob_label argument based on multiclass status
        average, pos_label = Metrics._configure_multiclass_parameters(y_true, kwargs, 'F1_SCORE')

        kwargs.pop("average", None)
        kwargs.pop("pos_label", None)

        try:
            return metrics.f1_score(y_true, y_pred, average=average, pos_label=pos_label, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `F1_SCORE` {str(e)}")

    @staticmethod
    def mse(y_true: Union[np.ndarray, list],
            y_pred: Union[np.ndarray, list],
            **kwargs):
        """
        Evaluate the mean squared error.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,),
            default=’uniform_average’, optional) Defines aggregating of multiple output values. Array-like value
            defines weights used to average errors.
            - squared (bool, default=True, optional)
            If True returns MSE value, if False returns RMSE value.
        Returns:
            - sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average',
            squared=True) score (float or ndarray of floats)
        """

        # Set multiouput as raw_values is it is not defined by researcher
        multi_output = kwargs.get('multioutput', 'raw_values')
        kwargs.pop('multioutput', None)

        try:
            return metrics.mean_squared_error(y_true, y_pred,  multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `MEAN_SQUARED_ERROR`"
                                       f" {str(e)}")

    @staticmethod
    def mae(y_true: Union[np.ndarray, list],
            y_pred: Union[np.ndarray, list],
            **kwargs):
        """
        Evaluate the mean absolute error.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,),
            default=’uniform_average’, optional) Defines aggregating of multiple output values. Array-like value
            defines weights used to average errors.
        Returns:
            - sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
            score (float or ndarray of floats)
        """
        # Set multiouput as raw_values is it is not defined by researcher
        multi_output = kwargs.get('multioutput', 'raw_values')
        kwargs.pop('multioutput', None)

        try:
            return metrics.mean_absolute_error(y_true, y_pred,  multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `MEAN_ABSOLUTE_ERROR`"
                                       f" {str(e)}")

    @staticmethod
    def explained_variance(y_true: Union[np.ndarray, list],
                           y_pred: Union[np.ndarray, list],
                           **kwargs):
        """
        Evaluate the accuracy score.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html]
        Args:
            - sample_weight (array-like of shape (n_samples,), default=None, optional)
            Sample weights.
            - multioutput ({‘raw_values’, ‘uniform_average’, ‘variance_weighted’} or array-like of shape (n_outputs,),
            default=’uniform_average’, optional) Defines aggregating of multiple output values. Array-like value
            defines weights used to average errors.
        Returns:
            - sklearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None,
            multioutput='uniform_average') score (float or ndarray of floats)
        """

        # Set multiouput as raw_values is it is not defined by researcher
        multi_output = kwargs.get('multioutput', 'raw_values')
        kwargs.pop('multioutput', None)

        try:
            return metrics.explained_variance_score(y_true, y_pred, multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `EXPLAINED_VARIANCE`"
                                       f" {str(e)}")

    @staticmethod
    def _configure_y_true_pred_(y_true: Union[np.ndarray, list],
                                y_pred: Union[np.ndarray, list],
                                metric: MetricTypes):
        """
        Method for configuring y_true and y_pred array to make them compatible for metric functions. It
        guarantees that the y_true and y_pred will be in same shape.

        Args:
            y_true (np.ndarray): True values of test dataset
            y_pred (np.ndarray): Predicted values
            metric (MetricTypes): Metric that is going to be used for evaluation
        """

        # Squeeze array [[1],[2],[3]] to [1,2,3]
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)

        if len(y_pred) != len(y_true):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Predictions or true values should have"
                                       f"equal number of samples, {len(y_true)}, {len(y_pred)}")

        # Get shape of the prediction should be 1D or 2D array
        shape_y_pred = y_pred.shape
        shape_y_true = y_true.shape

        # Shape of the prediction array should be (samples, outputs) or (samples, )
        if len(shape_y_pred) > 2 or len(shape_y_true) > 2:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Predictions or true values are not in "
                                       f"supported shape {y_pred.shape}, `{y_true.shape}`, should be 1D or 2D "
                                       f"list/array. If it isa special case,  please consider creating a custom "
                                       f"`testing_step` method in training plan")

        if Metrics._is_array_of_str(y_pred) != Metrics._is_array_of_str(y_true):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Predicted values and true values are not in"
                                       f"different types `int` and `str`")

        if Metrics._is_array_of_str(y_pred):
            if metric.metric_category() is _MetricCategory.REGRESSION:
                raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Can not apply metric `{metric.name}` "
                                           f"to non-numeric prediction results")
            return y_true, y_pred

        output_shape_y_pred = shape_y_pred[1] if len(shape_y_pred) == 2 else 0  # 0 for 1D array
        output_shape_y_true = shape_y_true[1] if len(shape_y_true) == 2 else 0  # 0 for 1D array

        if metric.metric_category() is _MetricCategory.CLASSIFICATION_LABELS:
            if output_shape_y_pred == 0 and output_shape_y_true == 0:
                # If y_pred contains labels as integer, do not use threshold cut
                if sum(y_pred) != sum([round(i) for i in y_pred]):
                    y_pred = np.where(y_pred > 0.5, 1, 0)

                if sum(y_true) != sum([round(i) for i in y_true]):
                    raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: True values are continuous, "
                                               f"classification metrics can't handle a mix of continuous and "
                                               f"binary targets")

            # If y_true is one 2D array and y_pred is 1D array
            # Example: y_true: [ [0,1], [1,0]] | y_pred : [0.1, 0.5]
            elif output_shape_y_pred == 0 and output_shape_y_true > 0:
                y_pred = np.where(y_pred > 0.5, 1, 0)
                y_true = np.argmax(y_true, axis=1)

            # If y_pred is 2D array where each array and y_true is 1D array of classes
            # Example: y_true: [0,1,1,2,] | y_pred : [[-0.2, 0.3, 0.5], [0.5, -1.2, 1,2 ], [0.5, -1.2, 1,2 ]]
            elif output_shape_y_pred > 0 and output_shape_y_true == 0:
                y_pred = np.argmax(y_pred, axis=1)

            # If y_pred and y_true is 2D array
            # Example: y_true: [ [0,1],[1,0]] | y_pred : [[-0.2, 0.3], [0.5, 1,2 ]]
            elif output_shape_y_pred > 0 and output_shape_y_true > 0:

                if output_shape_y_pred != output_shape_y_true:
                    raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Can not convert values to class labels, "
                                               f"shape of predicted and true values do not match.")
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_true, axis=1)

        elif metric.metric_category() is _MetricCategory.REGRESSION:
            if output_shape_y_pred != output_shape_y_true:
                raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: For the metric `{metric.name}` multiple "
                                           f"output regression is not supported")

        return y_true, y_pred

    @staticmethod
    def _is_array_of_str(list_: np.ndarray):
        """
        Method for checking whether list elements are of type string

        Args:
            list_ (np.ndarray): Numpy array that is going to be checked for types
        """

        if len(list_.shape) == 1:
            return True if isinstance(list_[0], str) else False
        else:
            return True if isinstance(list_[0][0], str) else False

    @staticmethod
    def _configure_multiclass_parameters(y_true, parameters, metric):

        average = parameters.get('average', 'binary')
        pos_label = parameters.get('pos_label', 1)
        # Check target variable is multi class or binary
        if len(np.unique(y_true)) > 2:
            average = parameters.get('average', 'weighted')
            logger.info(f'Actual/True values (y_true) has more than two levels, using multiclass `{average}` '
                        f'calculation for the metric {metric}')

        else:
            average = parameters.get('average', 'binary')
            # Alphabetically select first label as pos_label
            y_true_copy = copy(y_true)
            y_true_copy.copy()
            pos_label = y_true_copy[0] if isinstance(y_true[0], str) else parameters.get('pos_label', 1)

        return average, pos_label
