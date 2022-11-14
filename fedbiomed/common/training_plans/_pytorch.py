# coding: utf-8

"""Training Plan designed to wrap PyTorch `nn.Module` models."""

import functools
from typing import Any, Dict, Optional, Union

import declearn
import declearn.model.torch
import numpy as np
import torch

from fedbiomed.common.data import DataLoaderTypes
from fedbiomed.common.history_monitor import HistoryMonitor
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from ._base import TrainingPlan


class TorchTrainingPlan(TrainingPlan):
    """Base class for training plans wrapping `torch.nn.Module` models.

    All concrete torch training plans inheriting this class should implement:
        * the `training_data` method:
            to define how to set up the `fedbiomed.data.DataManager`
            wrapping the training (and, by split, validation) data
        * (opt.) the `testing_step` method:
            to override the evaluation behavior and compute
            a batch-wise (set of) metric(s)

    Attributes:
        model: declearn Model instance wrapping the model being trained.
        optim: declearn Optimizer in charge of node-side optimization.
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    _model_cls=declearn.model.torch.TorchModel
    _data_type=DataLoaderTypes.TORCH

    def __init__(
            self,
            model: Union[torch.nn.Module, Dict[str, Any]],
            optim: Union[declearn.optimizer.Optimizer, Dict[str, Any]],
            loss: Optional[torch.nn.Module] = None,
            **kwargs: Any
        ) -> None:
        """Construct the torch training plan.

        Args:
            model: Base `torch.nn.Module` object to be interfaced using
                a declearn `TorchModel`, or config dict of the latter.
            optim: declearn.optimizer.Optimizer instance of config dict.
            loss: Optional `torch.nn.Module` defining the model's loss
                (unused if `model` is a config dict).
        """
        super().__init__(model, optim, loss=loss, **kwargs)
        self._device = "cpu"  # name of the device backing train computations

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional[HistoryMonitor],
            before_train: bool
        ) -> None:
        try:
            model = getattr(self.model, "_model")  # type: torch.nn.Module
            model.eval()  # pytorch switch for model inference-mode
            super().testing_routine(
                metric, metric_args, history_monitor, before_train
            )
        finally:
            model.train()  # restore training behaviors

    def predict(
            self,
            data: torch.Tensor,
        ) -> np.ndarray:
        """Return model predictions for a given batch of input features.

        This method is called as part of `testing_routine`, to compute
        predictions based on which evaluation metrics are computed. It
        will however be skipped if a `testing_step` method is attached
        to the training plan, than wraps together a custom routine to
        compute an output metric directly from a (data, target) batch.

        !!! info "Note"
            The wrapped torch module is not checked to be in inference
            mode: this is to be handled from the caller (as is done as
            part of the `testing_routine` method).

        Args:
            data: torch.Tensor containing batched input features.

        Returns:
            Output predictions, converted to a numpy array (as per the
                `fedbiomed.common.metrics.Metrics` specs).
        """
        model = getattr(self.model, "_model")  # type: torch.nn.Module
        with torch.no_grad():
            pred = model(data)
            outp = pred.detach().numpy()  # type: np.ndarray
        return outp

    def training_routine(
            self,
            history_monitor: Optional[HistoryMonitor] = None,
            node_args: Optional[Dict[str, Any]] = None
        ) -> None:
        # Run the parent method, but ensure the model is moved back to CPU.
        try:
            super().training_routine(history_monitor, node_args)
        finally:
            model = getattr(self.model, "_model")  # type: torch.nn.Module
            model.cpu()
            torch.cuda.empty_cache()

    def _process_training_node_args(
            self,
            node_args: Dict[str, Any],
        ) -> None:
        # Optionally move the model to GPU.
        self._device = self._select_device(node_args)
        model = getattr(self.model, "_model")  # type: torch.nn.Module
        model.to(self._device)

    def _training_step(
            self,
            idx: int,
            inputs: torch.Tensor,
            target: torch.Tensor,
            record_loss: Optional[functools.partial] = None,
        ) -> None:
        # Ensure the data is backed on the same device as the model.
        inputs = inputs.to(self._device)
        target = target.to(self._device)
        super()._training_step(idx, inputs, target, record_loss)

    def _select_device(
            self,
            node_args: Dict[str, Any],
        ) -> str:
        """Select whether to use CPU of GPU device for training.

        Args:
            node_args: Node-specific training arguments to process.
                Note that researcher-set training arguments are also
                used, but can be overridden by the node-set ones.
        """
        if self._training_args is None:
            raise RuntimeError(
                "Cannot call `_select_device` on uninitialized training plan."
            )
        # Gather variables determining GPU usage choices.
        cuda_available = torch.cuda.is_available()
        gpu_serv = bool(self._training_args.get("use_gpu"))
        gpu_only = bool(node_args.get("gpu_only"))
        gpu_indx = node_args.get("gpu_num", None)  # type: Optional[int]
        gpu_node = (gpu_indx is not None) or bool(node_args.get("gpu"))
        # Decide whether GPU can and will be used.
        use_cuda = cuda_available and (gpu_only or (gpu_node and gpu_serv))
        # Warn about parameters getting overridden.
        if gpu_only:
            if not cuda_available:
                logger.error(
                    "Node wants to force model training on GPU, but no GPU "
                    "is available."
                )
            if not gpu_serv:
                logger.warning(
                    "Node forces model training on GPU, though it was not "
                    "requested by researcher."
                )
        if gpu_serv and not use_cuda:
            logger.warning(
                "Node training model on CPU, though researcher requested GPU."
            )
        # Select backing device for training.
        device = "cuda" if use_cuda else "cpu"
        if use_cuda and gpu_indx is not None:
            if gpu_indx < torch.cuda.device_count():
                device = f"cuda:{gpu_indx}"
            else:
                logger.warning(f"Bad GPU index {gpu_indx}, using default GPU.")
        # Log about the decision (and parsed arguments) and return the device.
        logger.debug(
            f"Using device {device} for training "
            f"(cuda_available={cuda_available}, use_gpu={gpu_serv}, "
            f"gpu={gpu_node}, gpu_only={gpu_only}, gpu_num={gpu_indx})."
        )
        return device
