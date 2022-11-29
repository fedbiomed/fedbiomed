# coding: utf-8

"""SKLearnTrainingPlan subclasses for models implementing `partial_fit`."""

import functools
import sys
from abc import ABCMeta
from contextlib import contextmanager
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_plans import SKLearnTrainingPlan


__all__ = [
    "FedPerceptron",
    "FedSGDClassifier",
    "FedSGDRegressor",
]


@contextmanager
def capture_stdout() -> Iterator[List[str]]:
    """Context manager to capture console outputs (stdout).

    Returns:
        A list, empty at first, that will be populated with the line-wise
        strings composing the captured stdout upon exiting the context.
    """
    output = []  # type: List[str]
    stdout = sys.stdout
    str_io = StringIO()
    # Capture stdout outputs into the StringIO. Return yet-empty list.
    try:
        sys.stdout = str_io
        yield output
    # Restore sys.stdout, then parse captured outputs for loss values.
    finally:
        sys.stdout = stdout
        output.extend(str_io.getvalue().splitlines())


class SKLearnTrainingPlanPartialFit(SKLearnTrainingPlan, metaclass=ABCMeta):
    """Base SKLearnTrainingPlan for models implementing `partial_fit`."""

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self._model_cls, 'partial_fit'):
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB302.value}: SKLearnTrainingPlanPartialFit"
                "requires the target scikit-learn model class to expose a"
                "`partial_fit` method."
            )

    def _training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None
        ) -> None:
        """Backend training routine for scikit-learn models with `partial_fit`.

        Args:
            history_monitor (HistoryMonitor, None): optional HistoryMonitor
                instance, recording the loss value during training.
        """
        # Gather reporting parameters.
        report = False
        if (history_monitor is not None) and hasattr(self._model, "verbose"):
            report = True
            log_interval = self._training_args.get("log_interval", 10)
            loss_name = getattr(self._model, "loss", "")
            loss_name = "Loss" + (f" {loss_name}" if loss_name else "")
            record_loss = functools.partial(
                history_monitor.add_scalar,
                train=True,
                num_batches=len(self.training_data_loader),
                total_samples=len(self.training_data_loader.dataset)
            )
            verbose = self._model.get_params("verbose")
            self._model.set_params(verbose=1)
        # Iterate over epochs.
        for epoch in range(self._training_args.get("epochs", 1)):
            # Iterate over data batches.
            for idx, batch in enumerate(self.training_data_loader, start=1):
                inputs, target = batch
                loss = self._train_over_batch(inputs, target, report)
                # Optionally report on the batch training loss.
                if report and (idx % log_interval == 0) and not np.isnan(loss):
                    record_loss(
                        metric={loss_name: loss},
                        iteration=idx,
                        epoch=epoch,
                        batch_samples=len(inputs)
                    )
                    logger.debug(
                        f"Train Epoch: {epoch} "
                        f"Batch: {idx}/{record_loss.keywords['num_batches']}"
                        f"\tLoss: {loss:.6f}"
                    )

                if 0 < self._batch_maxnum <= idx:
                    logger.info(f'Reached {self._batch_maxnum} batches for this epoch, ignore remaining data')
                    break
        # Reset model verbosity to its initial value.
        if report:
            self._model.set_params(verbose=verbose)

    def _train_over_batch(
            self,
            inputs: np.ndarray,
            target: np.ndarray,
            report: bool
        ) -> float:
        """Perform gradient descent over a single data batch.

        This method also resets the n_iter_ attribute of the
        scikit-learn model, such that n_iter_ will always equal
        1 at the end of the execution.

        Args:
            inputs: 2D-array of batched input features.
            target: 2D-array of batched target labels.
            report: Whether to capture and parse the training
                loss printed out to the console by the scikit-learn
                model. If False, or if parsing fails, return a nan.
        """
        b_len = inputs.shape[0]
        # Gather start weights of the model and initialize zero gradients.
        param = {k: getattr(self._model, k) for k in self._param_list}
        grads = {k: np.zeros_like(v) for k, v in param.items()}
        # Iterate over the batch; accumulate sample-wise gradients (and loss).
        stdout = []  # type: List[List[str]]
        for idx in range(b_len):
            # Compute updated weights based on the sample. Capture loss prints.
            with capture_stdout() as console:
                self._model.partial_fit(inputs[idx:idx+1], target[idx])
            stdout.append(console)
            # Accumulate updated weights (weights + sum of gradients).
            # Reset the model's weights and iteration counter.
            for key in self._param_list:
                grads[key] += getattr(self._model, key)
                setattr(self._model, key, param[key])
            self._model.n_iter_ -= 1
        # Compute the batch-averaged updated weights and apply them.
        # Update the `param` values, and reset gradients to zero.
        for key in self._param_list:
            setattr(self._model, key, grads[key] / b_len)
        self._model.n_iter_ += 1
        # Optionally report the training loss over this batch.
        if report:
            try:
                return self._parse_batch_loss(stdout, inputs, target)
            except Exception as exc:
                msg = (
                    f"{ErrorNumbers.FB605.value}: error while parsing "
                    f"training losses from stdout: {exc}"
                )
                logger.error(msg)
        # Otherwise, return nan as a fill-in value.
        return float('nan')

    def _parse_batch_loss(
            self,
            stdout: List[List[str]],
            inputs: np.ndarray,
            target: np.ndarray
        ) -> float:
        """Parse logged loss values from captured stdout lines.

        Args:
            stdout: Captured stdout outputs from calling
                the model's partial fit, with one list per batched sample.
            inputs: Batched input features.
            target: Batched target labels.
        """
        values = [self._parse_sample_losses(sample) for sample in stdout]
        losses = np.array(values)
        return float(np.mean(losses))

    @staticmethod
    def _parse_sample_losses(
            stdout: List[str]
        ) -> List[float]:
        """Parse logged loss values from captured stdout lines."""
        losses = []  # type: List[float]
        for row in stdout:
            split = row.rsplit("loss: ", 1)
            if len(split) == 1:  # no "loss: " in the line
                continue
            try:
                losses.append(float(split[1]))
            except ValueError as exc:
                logger.error(f"Value error during monitoring: {exc}")
        return losses


class FedSGDRegressor(SKLearnTrainingPlanPartialFit):
    """Fed-BioMed training plan for scikit-learn SGDRegressor models."""

    _model_cls = SGDRegressor
    _model_dep = (
        "from sklearn.linear_model import SGDRegressor",
        "from fedbiomed.common.training_plans import FedSGDRegressor"
    )

    def __init__(self) -> None:
        """Initialize the sklearn SGDRegressor training plan."""
        super().__init__()
        self._is_regression = True

    def set_init_params(self) -> None:
        """Initialize the model's trainable parameters."""
        init_params = {
            'intercept_': np.array([0.]),
            'coef_': np.array([0.] * self._model_args['n_features'])
        }
        self._param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self._model, key, val)

    def get_learning_rate(self) -> List[float]:
        return self._model.eta0


class FedSGDClassifier(SKLearnTrainingPlanPartialFit):
    """Fed-BioMed training plan for scikit-learn SGDClassifier models."""

    _model_cls = SGDClassifier
    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedSGDClassifier"
    )

    def __init__(self) -> None:
        """Initialize the sklearn SGDClassifier training plan."""
        super().__init__()
        self._is_classification = True

    def set_init_params(self) -> None:
        """Initialize the model's trainable parameters."""
        # Set up zero-valued start weights, for binary of multiclass classif.
        n_classes = self._model_args["n_classes"]
        if n_classes == 2:
            init_params = {
                "intercept_": np.zeros((1,)),
                "coef_": np.zeros((1, self._model_args["n_features"]))
            }
        else:
            init_params = {
                "intercept_": np.zeros((n_classes,)),
                "coef_": np.zeros((n_classes, self._model_args["n_features"]))
            }
        # Assign these initialization parameters and retain their names.
        self._param_list = list(init_params.keys())
        for key, val in init_params.items():
            setattr(self._model, key, val)
        # Also initialize the "classes_" slot with unique predictable labels.
        # FIXME: this assumes target values are integers in range(n_classes).
        setattr(self._model, "classes_", np.arange(n_classes))

    def get_learning_rate(self) -> List[float]:
        return self._model.eta0

    def _parse_batch_loss(
            self,
            stdout: List[List[str]],
            inputs: np.ndarray,
            target: np.ndarray
        ) -> float:
        """Parse logged loss values from captured stdout lines."""
        # Delegate binary classification case to parent class.
        if self._model_args["n_classes"] == 2:
            return super()._parse_batch_loss(stdout, inputs, target)
        # Handle multilabel classification case.
        # Compute and batch-average sample-wise label-wise losses.
        values = [self._parse_sample_losses(sample) for sample in stdout]
        losses = np.array(values).mean(axis=0)
        # Compute the support-weighted average of label-wise losses.
        # NOTE: this assumes a (n, 1)-shaped targets array.
        classes = getattr(self._model, "classes_")
        support = (target == classes).sum(axis=0)
        return float(np.average(losses, weights=support))


class FedPerceptron(FedSGDClassifier):
    """Fed-BioMed training plan for scikit-learn Perceptron models.

    This class inherits from FedSGDClassifier, and forces the wrapped
    scikit-learn SGDClassifier model to use a "perceptron" loss, that
    makes it equivalent to an actual scikit-learn Perceptron model.
    """

    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedPerceptron"
    )

    def __init__(self) -> None:
        """Class constructor."""
        super().__init__()
        self._model.set_params(loss="perceptron")

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
        ) -> None:
        model_args["loss"] = "perceptron"
        super().post_init(model_args, training_args)
