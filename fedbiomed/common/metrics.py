# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Provide test metrics, both MetricTypes to use in TrainingArgs but also calculation routines.
"""


from copy import copy
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from fedbiomed.common.constants import _BaseEnum, ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedMetricError


class _MetricCategory(_BaseEnum):

    CLASSIFICATION_LABELS = 0  # return labels
    REGRESSION = 2


class MetricTypes(_BaseEnum):
    """List of Performance metrics used to evaluate the model."""

    ACCURACY = (0, _MetricCategory.CLASSIFICATION_LABELS)
    F1_SCORE = (1, _MetricCategory.CLASSIFICATION_LABELS)
    PRECISION = (2, _MetricCategory.CLASSIFICATION_LABELS)
    RECALL = (3, _MetricCategory.CLASSIFICATION_LABELS)

    MEAN_SQUARE_ERROR = (4, _MetricCategory.REGRESSION)
    MEAN_ABSOLUTE_ERROR = (5, _MetricCategory.REGRESSION)
    EXPLAINED_VARIANCE = (6, _MetricCategory.REGRESSION)

    def __init__(self, idx: int, metric_category: _MetricCategory) -> None:
        self._idx = idx
        self._metric_category = metric_category

    def metric_category(self) -> _MetricCategory:
        return self._metric_category

    @staticmethod
    def get_all_metrics() -> List[str]:
        return [metric.name for metric in MetricTypes]

    @staticmethod
    def get_metric_type_by_name(metric_name: str):
        for metric in MetricTypes:
            if metric.name == metric_name:
                return metric


class Metrics(object):
    """Class of performance metrics used in validation evaluation."""

    def __init__(self):
        """Constructs metric class with provided metric types: metric function

        Attrs:
            metrics: Provided metrics in form of `{ MetricTypes : skleran.metrics }`
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
                 **kwargs: dict) -> Union[int, float]:
        """Perform evaluation based on given metric.

        This method configures given y_pred and y_true to make them compatible with default evaluation methods.

        Args:
            y_true: True values
            y_pred: Predicted values
            metric: An instance of MetricTypes to chose metric that will be used for evaluation
            **kwargs: The arguments specifics to each type of metrics.

        Returns:
            Result of the evaluation function

        Raises:
            FedbiomedMetricError: in case of invalid metric, y_true and y_pred types
        """
        if not isinstance(metric, MetricTypes):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Metric should instance of `MetricTypes`")

        if y_true is not None and not isinstance(y_true, (np.ndarray, list)):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_true` should an instance "
                                       f"of `np.ndarray`, but got {type(y_true)} ")

        if y_pred is not None and not isinstance(y_pred, (np.ndarray, list)):
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: The argument `y_pred` should an instance "
                                       f"of `np.ndarray`, but got {type(y_pred)} ")

        y_true, y_pred = self._configure_y_true_pred_(y_true=y_true, y_pred=y_pred, metric=metric)
        result = self.metrics[metric.name](y_true, y_pred, **kwargs)

        return result

    @staticmethod
    def accuracy(y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list],
                 **kwargs: dict) -> float:
        """ Evaluate the accuracy score

        Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.accuracy_score`][sklearn.metrics.accuracy_score]

        Returns:
            Accuracy score

        Raises:
            FedbiomedMetricError: raised if above sklearn method raises
        """

        try:
            y_true, y_pred, _, _ = Metrics._configure_multiclass_parameters(y_true, y_pred, kwargs, 'ACCURACY')
            return metrics.accuracy_score(y_true, y_pred, **kwargs)
        except Exception as e:
            msg = ErrorNumbers.FB611.value + " Exception raised from SKLEARN metrics: " + str(e)
            raise FedbiomedMetricError(msg)

    @staticmethod
    def precision(y_true: Union[np.ndarray, list],
                  y_pred: Union[np.ndarray, list],
                  **kwargs: dict) -> float:
        """Evaluate the precision score
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html]

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.precision_score`][sklearn.metrics.precision_score]

        Returns:
            precision (float, or array of float of shape (n_unique_labels,))

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """
        # Get average and pob_label argument based on multiclass status
        y_true, y_pred, average, pos_label = Metrics._configure_multiclass_parameters(y_true,
                                                                                      y_pred,
                                                                                      kwargs,
                                                                                      'PRECISION')

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
               **kwargs: dict) -> float:
        """Evaluate the recall.
        [source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html]

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.recall_score`][sklearn.metrics.recall_score]

        Returns:
            recall (float (if average is not None) or array of float of shape (n_unique_labels,))

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """

        # Get average and pob_label argument based on multiclass status
        y_true, y_pred, average, pos_label = Metrics._configure_multiclass_parameters(y_true, y_pred, kwargs, 'RECALL')

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
                 **kwargs: dict) -> float:
        """Evaluate the F1 score.

        Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.f1_score`][sklearn.metrics.f1_score]

        Returns:
            f1_score (float or array of float, shape = [n_unique_labels])

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """

        # Get average and pob_label argument based on multiclass status
        y_true, y_pred, average, pos_label = Metrics._configure_multiclass_parameters(y_true,
                                                                                      y_pred,
                                                                                      kwargs,
                                                                                      'F1_SCORE')

        kwargs.pop("average", None)
        kwargs.pop("pos_label", None)

        try:
            return metrics.f1_score(y_true, y_pred, average=average, pos_label=pos_label, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `F1_SCORE` {str(e)}")

    @staticmethod
    def mse(y_true: Union[np.ndarray, list],
            y_pred: Union[np.ndarray, list],
            **kwargs: dict) -> float:
        """Evaluate the mean squared error.

        Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.mean_squared_error`][sklearn.metrics.mean_squared_error]

        Returns:
            MSE score (float or ndarray of floats)

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """

        # Set multiouput as raw_values is it is not defined by researcher
        if len(y_true.shape) > 1:
            multi_output = kwargs.get('multioutput', 'raw_values')
        else:
            multi_output = None

        kwargs.pop('multioutput', None)

        try:
            return metrics.mean_squared_error(y_true, y_pred, multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `MEAN_SQUARED_ERROR`"
                                       f" {str(e)}")

    @staticmethod
    def mae(y_true: Union[np.ndarray, list],
            y_pred: Union[np.ndarray, list],
            **kwargs: dict) -> float:
        """Evaluate the mean absolute error.

        Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.mean_absolute_error`][sklearn.metrics.mean_absolute_error]

        Returns:
            MAE score (float or ndarray of floats)

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """
        # Set multiouput as raw_values is it is not defined by researcher
        if len(y_true.shape) > 1:
            multi_output = kwargs.get('multioutput', 'raw_values')
        else:
            multi_output = None

        kwargs.pop('multioutput', None)

        try:
            return metrics.mean_absolute_error(y_true, y_pred, multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(
                f"{ErrorNumbers.FB611.value}: Error during calculation of `MEAN_ABSOLUTE_ERROR`"
                f" {str(e)}") from e

    @staticmethod
    def explained_variance(y_true: Union[np.ndarray, list],
                           y_pred: Union[np.ndarray, list],
                           **kwargs: dict) -> float:
        """Evaluate the Explained variance regression score.

        Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html]

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Extra arguments from [`sklearn.metrics.explained_variance_score`]
                [sklearn.metrics.explained_variance_score]

        Returns:
            EV score (float or ndarray of floats)

        Raises:
            FedbiomedMetricError: raised if above sklearn method for computing precision raises
        """

        # Set multiouput as raw_values is it is not defined by researcher
        if len(y_true.shape) > 1:
            multi_output = kwargs.get('multioutput', 'raw_values')
        else:
            multi_output = None

        kwargs.pop('multioutput', None)

        try:
            return metrics.explained_variance_score(y_true, y_pred, multioutput=multi_output, **kwargs)
        except Exception as e:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Error during calculation of `EXPLAINED_VARIANCE`"
                                       f" {str(e)}")

    @staticmethod
    def _configure_y_true_pred_(y_true: Union[np.ndarray, list],
                                y_pred: Union[np.ndarray, list],
                                metric: MetricTypes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for configuring y_true and y_pred array to make them compatible for metric functions. It
        guarantees that the y_true and y_pred will have same shape.

        Args:
            y_true: True values (lablels) of validation dataset
            y_pred: Predicted values
            metric: Metric that is going to be used for evaluation

        Returns:
            Tuple of y_true and y_pred as;
                - y_true: updated labels of dataset
                - y_pred: updated prediction values

        Raises:
            FedbiomedMetricError: Invalid shape of `y_true` or `y_pred`
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Special case to support batch_size=1: issue #1013
        if y_true.ndim == 2 and len(y_true[0]) == 1:
            y_true = np.squeeze(y_true)

        if y_pred.ndim == 2 and len(y_pred[0]) == 1:
            y_pred = np.squeeze(y_pred)
        # ------------------------------------------------

        if y_pred.ndim == 0:
            y_pred = y_pred.reshape((1,))
        if y_true.ndim == 0:
            y_true = y_true.reshape((1,))

        if y_pred.shape[0] != y_true.shape[0]:
            raise FedbiomedMetricError(f"{ErrorNumbers.FB611.value}: Predictions and true values should have"
                                       f"equal number of samples, but got y_true = {len(y_true)}, and"
                                       f" y_pred= {len(y_pred)}")

        # Get shape of the prediction should be 1D or 2D array
        shape_y_pred = y_pred.shape
        shape_y_true = y_true.shape

        # Shape of the prediction array should be (samples, outputs) or (samples, )
        if len(shape_y_pred) > 2 or len(shape_y_true) > 2:
            raise FedbiomedMetricError(
                f"{ErrorNumbers.FB611.value}: Predictions or true values are not in "
                f"supported shape {y_pred.shape}, `{y_true.shape}`, should be 1D or 2D "
                f"list/array. If it isa special case,  please consider creating a custom "
                f"`testing_step` method in training plan")

        if Metrics._is_array_of_str(y_pred) != Metrics._is_array_of_str(y_true):
            raise FedbiomedMetricError(
                f"{ErrorNumbers.FB611.value}: Predicted values and true values have "
                f"different types `int` and `str`")

        if Metrics._is_array_of_str(y_pred):
            if metric.metric_category() is _MetricCategory.REGRESSION:
                raise FedbiomedMetricError(
                    f"{ErrorNumbers.FB611.value}: Can not apply metric `{metric.name}` "
                    f"to non-numeric prediction results")
            return y_true, y_pred

        output_shape_y_pred = shape_y_pred[1] if len(shape_y_pred) == 2 else 0  # 0 for 1D array
        output_shape_y_true = shape_y_true[1] if len(shape_y_true) == 2 else 0  # 0 for 1D array
        if metric.metric_category() is _MetricCategory.CLASSIFICATION_LABELS:
            if output_shape_y_pred == 0 and output_shape_y_true == 0:
                # If y_pred contains labels as integer, do not use threshold cut
                if sum(y_pred) != sum(round(i) for i in y_pred):
                    y_pred = np.where(y_pred > 0.5, 1, 0)

                if sum(y_true) != sum(round(i) for i in y_true):
                    y_true = np.where(y_true > 0.5, 1, 0)
                    logger.warning(
                        f"Target data seems to be a regression, metric {metric.name} might "
                        "not be appropriate", broadcast=True)

            # If y_true is one 2D array and y_pred is 1D array
            # Example: y_true: [ [0,1], [1,0]] | y_pred : [0.1, 0.5]
            elif output_shape_y_pred == 0 and output_shape_y_true > 0:
                y_pred = np.where(y_pred > 0.5, 1, 0)
                y_true = np.argmax(y_true, axis=1)

            # If y_pred is 2D array where each array and y_true is 1D array of classes
            # Example: y_true: [0,1,1,2,] | y_pred : [[-0.2, 0.3, 0.5],
            # [0.5, -1.2, 1,2 ], [0.5, -1.2, 1,2 ]]
            elif output_shape_y_pred > 0 and output_shape_y_true == 0:
                y_pred = np.argmax(y_pred, axis=1)

            # If y_pred and y_true is 2D array
            # Example: y_true: [ [0,1],[1,0]] | y_pred : [[-0.2, 0.3], [0.5, 1,2 ]]
            elif output_shape_y_pred > 0 and output_shape_y_true > 0:
                if output_shape_y_pred != output_shape_y_true:
                    raise FedbiomedMetricError(
                        f"{ErrorNumbers.FB611.value}: Can not convert values to class labels, "
                        f"shapes of predicted and true values do not match.")
                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_true, axis=1)

        return y_true, y_pred

    @staticmethod
    def _is_array_of_str(list_: np.ndarray) -> bool:
        """Checks whether list elements are of type string

        Args:
            list_: Numpy array that is going to be checked for types

        Returns:
            True if elements are of type string or False vice versa
        """

        if len(list_.shape) == 1:
            return isinstance(list_[0], str)

        return isinstance(list_[0][0], str)

    @staticmethod
    def _configure_multiclass_parameters(y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         parameters: Dict[str, Any],
                                         metric: str) -> Tuple[np.ndarray, np.ndarray, str, int]:
        """Re-format data giving the size of y_true and y_pred,

        In order to compute classification validation metric. If multiclass dataset, returns one hot encoding dataset.
        else returns binary class dataset

        Args:
            y_true: labels of data needed for classification
            y_pred: predicted values from the model. y_pred should tend towards y_true, for an optimal classification.
            parameters: parameters for the metric (metric_testing_args).

                Nota: If entry 'average' is missing, defaults to 'weighted' if multiclass dataset, defaults to 'binary'
                otherwise. If entry 'pos_label' is missing, defaults to 1.
            metric: name of the Metric

        Returns:
            As `y_true`, reformatted y_true
            As `y_pred`, reformatted y_pred
            As `average`, method name to be used in `average` argument (in the sklearn metric)
            As `pos_label`, label in dataset chosen to be considered as the positive dataset. Needed
                for computing Precision, recall, ...
        """

        pos_label = parameters.get('pos_label', None)
        # Check target variable is multi class or binary
        if len(np.unique(y_true)) > 2:
            average = parameters.get('average', 'weighted')

            encoder = OneHotEncoder()
            y_true = np.expand_dims(y_true, axis=1)
            y_pred = np.expand_dims(y_pred, axis=1)

            if not np.array_equal(np.unique(y_true), np.unique(y_pred)):
                y_true_and_pred = np.concatenate([y_true, y_pred], axis=0)
                encoder.fit(y_true_and_pred)
            else:
                encoder.fit(y_true)

            y_true = encoder.transform(y_true).toarray()
            y_pred = encoder.transform(y_pred).toarray()
        else:
            average = parameters.get('average', 'binary')
            # Alphabetically select first label as pos_label
            if pos_label is None:
                y_true_copy = copy(y_true)
                y_true_copy.sort()
                pos_label = y_true_copy[0]

        return y_true, y_pred, average, pos_label
