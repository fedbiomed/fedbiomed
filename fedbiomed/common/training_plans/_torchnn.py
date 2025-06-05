# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""TrainingPlan definition for the pytorch deep learning framework."""

from abc import ABCMeta, abstractmethod

from typing import Any, Dict, List, Tuple, OrderedDict, Optional, Union, Iterator



import torch

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.models import TorchModel
from fedbiomed.common.optimizers.generic_optimizers import BaseOptimizer, OptimizerBuilder
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer
from fedbiomed.common.privacy import DPController
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant
from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan
from fedbiomed.common.utils import get_method_spec


ModelInputType = Union[torch.Tensor, Dict, List, Tuple]


class TorchTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Implements  TrainingPlan for torch NN framework

    An abstraction over pytorch module to run pytorch models and scripts on node side. Researcher model (resp. params)
    will be:

    1. saved  on a '*.py' (resp. '*.mpk') files,
    2. uploaded on a HTTP server (network layer),
    3. then Downloaded from the HTTP server on node side,
    4. finally, read and executed on node side.

    Researcher must define/override:
    - a `training_data()` function
    - a `training_step()` function

    Researcher may have to add extra dependencies/python imports, by using `init_dependencies` method.

    Attributes:
        dataset_path: The path that indicates where dataset has been stored
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
        correction_state: an OrderedDict of {'parameter name': torch.Tensor} where the keys correspond to the names of
            the model parameters contained in self._model.named_parameters(), and the values correspond to the
            correction to be applied to that parameter.

    !!! info "Notes"
        The trained model may be exported via the `export_model` method,
        resulting in a dump file that may be reloded using `torch.save`
        outside of Fed-BioMed.
    """

    def __init__(self):
        """ Construct training plan """

        super().__init__()

        self.__type = TrainingPlans.TorchTrainingPlan

        # Differential privacy support
        self._dp_controller: Optional[DPController] = None
        self._optimizer: Union[BaseOptimizer, None] = None
        self._model: Union[TorchModel, None] = None

        self._use_gpu: bool = False
        self._share_persistent_buffers = None

        self._batch_maxnum: int = 100
        self._fedprox_mu: Optional[float] = None
        self._log_interval: int = 10
        self._epochs: int = 1
        self._dry_run = False
        self._num_updates: Optional[int] = None

        self.correction_state: OrderedDict = OrderedDict()
        self.aggregator_name: str = None

        # TODO : add random seed init
        # self.random_seed_params = None
        # self.random_seed_shuffling_data = None

        # device to use: cpu/gpu
        # - all operations except training only use cpu
        # - researcher doesn't request to use gpu by default
        self._device_init: str = "cpu"
        self._device = self._device_init

        # list dependencies of the model
        self._add_dependency(["import torch",
                             "import torch.nn as nn",
                             "import torch.nn.functional as F",
                             "from fedbiomed.common.training_plans import TorchTrainingPlan",
                             "from fedbiomed.common.data import DataManager",
                             "from fedbiomed.common.constants import ProcessTypes",
                             "from torch.utils.data import DataLoader",
                             "from torchvision import datasets, transforms"
                             ])

        # Aggregated model parameters
        #self._init_params: List[torch.Tensor] = None

        # Add dependencies
        self._configure_dependencies()

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
            initialize_optimizer: bool = True
    ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
            initialize_optimizer: If True, configures optimizer. It has to be True for node
                side configuration to prepare optimizer for the training.
        Raises:
            FedbiomedTrainingPlanError: If the provided arguments do not
                match expectations, or if the optimizer, model and dependencies
                configuration goes wrong.
        """
        super().post_init(model_args, training_args, aggregator_args)
        # Assign scalar attributes.
        self._use_gpu = self._training_args.get('use_gpu')
        self._batch_maxnum = self._training_args.get('batch_maxnum')
        self._log_interval = self._training_args.get('log_interval')
        self._epochs = self._training_args.get('epochs')
        self._num_updates = self._training_args.get('num_updates', 1)
        self._dry_run = self._training_args.get('dry_run')
        self._share_persistent_buffers = training_args.get('share_persistent_buffers', True)
        # Set random seed (Pytorch-specific)
        rseed = training_args['random_seed']

        if rseed is not None:
            torch.manual_seed(rseed)
        # Optionally set up differential privacy.
        self._dp_controller = DPController(training_args.dp_arguments() or None)
        # Configure aggregator-related arguments
        # TODO: put fedprox mu inside strategy_args
        self._fedprox_mu = self._training_args.get('fedprox_mu')
        self.set_aggregator_args(aggregator_args or {})
        # Configure the model and optimizer.

        self._configure_model_and_optimizer(initialize_optimizer)

    @abstractmethod
    def init_model(self):
        """Abstract method where model should be defined."""

    @abstractmethod
    def training_step(self):
        """Abstract method, all subclasses must provide a training_step."""

    @abstractmethod
    def training_data(self):
        """Abstract method to return training data"""

    def model(self) -> Optional[torch.nn.Module]:
        if self._model is None:
            return None
        return self._model.model

    def update_optimizer_args(self) -> Dict:
        """
        Updates `_optimizer_args` variable. Can prove useful
        to retrieve optimizer parameters after having trained a
        model, parameters which may have changed during training (eg learning rate).

        Updated arguments:
         - learning_rate

        Returns:
            Dict: updated `_optimizer_args`
        """
        if self._optimizer_args is None:
            self._optimizer_args = {}
        if self.aggregator_name is not None and self.aggregator_name.lower() == "scaffold":
            self._optimizer_args['lr'] = self._optimizer.get_learning_rate()
        return self._optimizer_args

    def optimizer_args(self) -> Dict[str, Any]:
        """Retrieves optimizer arguments

        Returns:
            Optimizer arguments
        """
        self.update_optimizer_args()  # update `optimizer_args` (eg after training)
        return super().optimizer_args()

    def initial_parameters(self) -> Dict:
        """Returns initial parameters without DP or training applied

        Returns:
            State dictionary of torch Module
        """
        return self._model.init_params

    def init_optimizer(self) -> Union[FedOptimizer, torch.optim.Optimizer]:
        """Abstract method for declaring optimizer by default """
        try:
            self._optimizer = torch.optim.Adam(self._model.model.parameters(), **self._optimizer_args)
        except AttributeError as e:
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Invalid argument for default "
                                             f"optimizer Adam. Error: {e}") from e

        return self._optimizer

    def type(self) -> TrainingPlans.TorchTrainingPlan:
        """ Gets training plan type"""
        return self.__type

    def _configure_model_and_optimizer(
        self,
        initialize_optimizer: bool = True
    ):
        """Configures model and optimizer before training

        Args:
            initialize_optimizer: If True configures optimizer.
        """

        # Message to format for unexpected argument definitions in special methods
        method_error = \
            ErrorNumbers.FB605.value + ": Special method `{method}` has more than one argument: {keys}. This method " \
                                       "can not have more than one argument/parameter (for {prefix} arguments) or " \
                                       "method can be defined without argument and `{alternative}` can be used for " \
                                       "accessing {prefix} arguments defined in the experiment."

        # Get model defined by user -----------------------------------------------------------------------------
        init_model_spec = get_method_spec(self.init_model)
        if not init_model_spec:
            model = self.init_model()
        elif len(init_model_spec.keys()) == 1:
            model = self.init_model(self._model_args)
        else:
            raise FedbiomedTrainingPlanError(method_error.format(prefix="model",
                                                                 method="init_model",
                                                                 keys=list(init_model_spec.keys()),
                                                                 alternative="self.model_args()"))

        # Validate and fix model
        model = self._dp_controller.validate_and_fix_model(model)
        self._model = TorchModel(model)

        # Get optimizer defined by researcher ------------------------
        # FIXME: This is implemented to solve the issue while setting model
        # arguments before setting training arguments on the researcher side (#1048).
        # the execution of init_optimizer during post init requires to have
        # optimizer_args set. However, this happens only on the researcher side
        # where optimizer of training plan is not used (see issue #1048).
        # Therefore, this fix adds new argument initialize_optimizer to set it False
        # on the researcher.
        if initialize_optimizer:
            init_optim_spec = get_method_spec(self.init_optimizer)
            if not init_optim_spec:
                optimizer = self.init_optimizer()
            elif len(init_optim_spec.keys()) == 1:
                optimizer = self.init_optimizer(self._optimizer_args)
            else:
                raise FedbiomedTrainingPlanError(
                    method_error.format(
                        prefix="optimizer",
                        method="init_optimizer",
                        keys=list(init_optim_spec.keys()),
                        alternative="self.optimizer_args()"))

            # Validate optimizer
            optim_builder = OptimizerBuilder()
            #  build the optimizer wrapper
            self._optimizer = optim_builder.build(self.__type, self._model, optimizer)

    def _set_device(self, use_gpu: Union[bool, None], node_args: dict):
        """Set device (CPU, GPU) that will be used for training, based on `node_args`

        Args:
            use_gpu: researcher requests to use GPU (or not)
            node_args: command line arguments for node
        """

        # set default values for node args
        if 'gpu' not in node_args:
            node_args['gpu'] = False
        if 'gpu_num' not in node_args:
            node_args['gpu_num'] = None
        if 'gpu_only' not in node_args:
            node_args['gpu_only'] = False

        # Training uses gpu if it exists on node and
        # - either proposed by node + requested by training plan
        # - or forced by node
        cuda_available = torch.cuda.is_available()
        if use_gpu is None:
            use_gpu = self._use_gpu
        use_cuda = cuda_available and ((use_gpu and node_args['gpu']) or node_args['gpu_only'])

        if node_args['gpu_only'] and not cuda_available:
            logger.error('Node wants to force model training on GPU, but no GPU is available',
                         broadcast=True)
        if use_cuda and not use_gpu:
            logger.warning('Node enforces model training on GPU, though it is not requested by researcher',
                           broadcast=True)
        if not use_cuda and use_gpu:
            logger.warning('Node training model on CPU, though researcher requested GPU', broadcast=True)

        # Set device for training

        if use_cuda:
            if node_args['gpu_num'] is not None:
                if node_args['gpu_num'] in range(torch.cuda.device_count()):
                    self._device = "cuda:" + str(node_args['gpu_num'])
                else:
                    logger.warning(f"Bad GPU number {node_args['gpu_num']}, using default GPU",
                                   broadcast=True)
                    self._device = "cuda"
            else:
                self._device = "cuda"
            self._optimizer.send_to_device(True, node_args['gpu_num'])
        else:
            self._device = "cpu"
            self._optimizer.send_to_device(False)  # specific to declearn

        logger.debug(f"Using device {self._device} for training "
                     f"(cuda_available={cuda_available}, gpu={node_args['gpu']}, "
                     f"gpu_only={node_args['gpu_only']}, "
                     f"use_gpu={use_gpu}, gpu_num={node_args['gpu_num']})")

    def send_to_device(self,
                       to_send: Union[torch.Tensor, list, tuple, dict],
                       device: torch.device
                       ):
        """Send inputs to correct device for training.

        Recursively traverses lists, tuples and dicts until it meets a torch Tensor, then sends the Tensor
        to the specified device.

        Args:
            to_send: the data to be sent to the device.
            device: the device to send the data to.

        Raises:
           FedbiomedTrainingPlanError: when to_send is not the correct type
        """
        if isinstance(to_send, torch.Tensor):
            return to_send.to(device)
        elif isinstance(to_send, dict):
            return {key: self.send_to_device(val, device) for key, val in to_send.items()}
        elif isinstance(to_send, tuple):
            return tuple(self.send_to_device(d, device) for d in to_send)
        elif isinstance(to_send, list):
            return [self.send_to_device(d, device) for d in to_send]
        else:
            raise FedbiomedTrainingPlanError(f'{ErrorNumbers.FB310.value} cannot send data to device. '
                                             f'Data must be a torch Tensor or a list, tuple or dict '
                                             f'ultimately containing Tensors.')

    def training_routine(self,
                         history_monitor: Any = None,
                         node_args: Union[dict, None] = None,
                         ) -> int:
        # FIXME: add betas parameters for ADAM solver + momentum for SGD
        # FIXME 2: remove parameters specific for validation specified in the
        # training routine
        """Training routine procedure.

        End-user should define;

        - a `training_data()` function defining how sampling / handling data in node's dataset is done. It should
            return a generator able to output tuple (batch_idx, (data, targets)) that is iterable for each batch.
        - a `training_step()` function defining how cost is computed. It should output loss values for backpropagation.

        Args:
            history_monitor: Monitor handler for real-time feed. Defined by the Node and can't be overwritten
            node_args: command line arguments for node. Can include:
                - `gpu (bool)`: propose use a GPU device if any is available. Default False.
                - `gpu_num (Union[int, None])`: if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available. Default None.
                - `gpu_only (bool)`: force use of a GPU device if any available, even if researcher
                    doesn't request for using a GPU. Default False.
        Returns:
            Total number of samples observed during the training.
        """

        #self.model().train()  # pytorch switch for training
        self._optimizer.init_training()
        # set correct type for node args
        node_args = {} if not isinstance(node_args, dict) else node_args

        # send all model to device, ensures having all the requested tensors
        self._set_device(self._use_gpu, node_args)
        self._model.send_to_device(self._device)

        # Run preprocess when everything is ready before the training
        self._preprocess()

        # # initial aggregated model parameters
        # self._init_params = deepcopy(list(self.model().parameters()))

        # DP actions
        self._optimizer, self.training_data_loader = \
            self._dp_controller.before_training(optimizer= self._optimizer, loader=self.training_data_loader)

        # set number of training loop iterations
        iterations_accountant = MiniBatchTrainingIterationsAccountant(self)

        # Training loop iterations
        for epoch in iterations_accountant.iterate_epochs():
            training_data_iter: Iterator = iter(self.training_data_loader)

            for batch_idx in iterations_accountant.iterate_batches():
                # retrieve data and target
                data, target = next(training_data_iter)

                # update accounting for number of observed samples
                batch_size = self._infer_batch_size(data)
                iterations_accountant.increment_sample_counters(batch_size)

                # handle training on accelerator devices
                data, target = self.send_to_device(data, self._device), self.send_to_device(target, self._device)

                # train this batch
                corrected_loss, loss = self._train_over_batch(data, target)

                # Reporting
                if iterations_accountant.should_log_this_batch():
                    # Retrieve reporting information: semantics differ whether num_updates or epochs were specified
                    num_samples, num_samples_max = iterations_accountant.reporting_on_num_samples()
                    num_iter, num_iter_max = iterations_accountant.reporting_on_num_iter()
                    epoch_to_report = iterations_accountant.reporting_on_epoch()

                    logger.debug('Train {}| '
                                 'Iteration {}/{} | '
                                 'Samples {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                                     f'Epoch: {epoch_to_report} ' if epoch_to_report is not None else '',
                                     num_iter,
                                     num_iter_max,
                                     num_samples,
                                     num_samples_max,
                                     100. * num_iter / num_iter_max,
                                     loss.item())
                                 )

                    # Send scalar values via general/feedback topic
                    if history_monitor is not None:
                        # the researcher only sees the average value of samples observed until now
                        history_monitor.add_scalar(metric={'Loss': loss.item()},
                                                   iteration=num_iter,
                                                   epoch=epoch_to_report,
                                                   train=True,
                                                   num_samples_trained=num_samples,
                                                   num_batches=num_iter_max,
                                                   total_samples=num_samples_max,
                                                   batch_samples=batch_size)

                # Handle dry run mode
                if self._dry_run:
                    self._model.send_to_device(self._device_init)
                    torch.cuda.empty_cache()
                    return iterations_accountant.num_samples_observed_in_total

        # release gpu usage as much as possible though:
        # - it should be done by deleting the object
        # - and some gpu memory remains used until process (cuda kernel ?) finishes

        self._model.send_to_device(self._device_init)
        torch.cuda.empty_cache()

        # # test (to be removed)
        # assert id(self._optimizer.model.model) == id(self._model.model)

        # assert id(self._optimizer.model) == id(self._model)

        # for (layer, val), (layer2, val2) in zip(self._model.model.state_dict().items(), self._optimizer.model.model.state_dict().items()):
        #     assert layer == layer2
        #     print(val, layer2)
        #     assert torch.isclose(val, val2).all()

        # for attributes, values in self._model.__dict__.items():
        #     print("ATTRIBUTES", values)
        #     assert values == getattr(self._optimizer.model, attributes)

        # for attributes, values in self._model.model.__dict__.items():
        #     print("ATTRIBUTES", values)
        #     assert values == getattr(self._optimizer.model.model, attributes)
        return iterations_accountant.num_samples_observed_in_total

    def _train_over_batch(self, data: ModelInputType, target: ModelInputType) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model over a single batch of data.

        This function handles all the torch-specific logic concerning model training, including backward propagation,
        aggregator-specific correction terms, and optimizer stepping.

        Args:
            data: the input data to the model
            target: the training labels

        Returns:
            corrected loss: the loss value used for backward propagation, including any correction terms
            loss: the uncorrected loss for reporting
        """
        # zero-out gradients
        self._optimizer.zero_grad()
        # FIXME: `self._optimizer.train()` is never called but should be.
        # FIXME 2: Should we move training process to `Optimizer` or `Model` class?

        # compute loss
        loss = self.training_step(data, target)  # raises an exception if not provided
        corrected_loss = torch.clone(loss)

        # If FedProx is enabled: use regularized loss function
        if self._fedprox_mu is not None:
            corrected_loss += float(self._fedprox_mu) / 2 * self.__norm_l2()

        # Run the backward pass to compute parameters' gradients
        corrected_loss.backward()

        # If Scaffold is used: apply corrections to the gradients
        if self.aggregator_name is not None and self.aggregator_name.lower() == "scaffold":
            for name, param in self.model().named_parameters():
                correction = self.correction_state.get(name)
                if correction is not None:
                    param.grad.sub_(correction.to(param.grad.device))

        # Have the optimizer collect, refine and apply gradients
        self._optimizer.step()

        return corrected_loss, loss

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional['HistoryMonitor'],
            before_train: bool
    ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        if not isinstance(self.model(), torch.nn.Module):
            msg = (
                f"{ErrorNumbers.FB320.value}: model should be a torch "
                f"nn.Module, but is of type {type(self.model())}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        try:

            with torch.no_grad():
                super().testing_routine(
                    metric, metric_args, history_monitor, before_train
                )
        finally:
            self.model().train()  # restore training behaviors

    def set_aggregator_args(self, aggregator_args: Dict[str, Any]):
        """Handles and loads aggregators arguments to received from the researcher

        Args:
            aggregator_args (Dict[str, Any]): dictionary mapping aggregator argument
                name with its value (eg 'aggregator_correction' with correction states)
        """

        self.aggregator_name = aggregator_args.get('aggregator_name', self.aggregator_name)
        # FIXME: this is too specific to Scaffold. Should be redesigned, or handled
        # by an aggregator handler that contains all keys for all strategies
        # implemented in fedbiomed
        # here we ae loading all args that have been sent from file exchange system
        for arg_name, aggregator_arg in aggregator_args.items():
            if arg_name == 'aggregator_correction':
                if not isinstance(aggregator_arg, dict):
                    raise FedbiomedTrainingPlanError(
                        f"{ErrorNumbers.FB309.value}: TorchTrainingPlan received "
                        "invalid 'aggregator_correction' aggregator args."
                    )
                self.correction_state = aggregator_arg

    def after_training_params(self, flatten: bool = False) -> Dict[str, torch.Tensor]:
        """Return the wrapped model's parameters for aggregation.

        This method returns a dict containing parameters that need to be
        reported back and aggregated in a federated learning setting.

        If the `postprocess` method exists (i.e. has been defined by end-users)
        it is called in the context of this method. DP-required adjustments are
        also set to happen as part of this method.

        If the researcher specified `share_persistent_buffers: False` in the
        training arguments, then we return only the output of
        [Model.get_weights][fedbiomed.common.models.TorchModel.get_weights],
        which considers only the trainable parameters.
        Otherwise, the default behaviour is to return the complete `state_dict`.

        Returns:
            The trained parameters to aggregate.
        """
        # Either include non-parameter buffers to the outputs or not.
        # Note: this is mostly about sharing statistics from BatchNorm layers.
        params = super().after_training_params()
        # Check whether postprocess method exists, and use it.
        if hasattr(self, 'postprocess'):
            logger.debug("running model.postprocess() method")
            try:
                params = self.postprocess(self._model.model.state_dict())  # Post process
            except Exception as e:
                raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605.value}: Error while running post-process "
                                                 f"{e}") from e

        # Run (optional) DP controller adjustments as well.
        params = self._dp_controller.after_training(params)
        if flatten:
            params = self._model.flatten(exclude_buffers=not self._share_persistent_buffers)
        return params

    def __norm_l2(self) -> float:
        """Regularize L2 that is used by FedProx optimization

        Returns:
            L2 norm of model parameters (before local training)
        """
        norm = 0

        for layer_name, init_coefs in self._model.init_params.items():
            current_model = self.model().get_parameter(layer_name)

            if current_model.requires_grad:
                norm += torch.linalg.norm(current_model - init_coefs) ** 2
        return norm
