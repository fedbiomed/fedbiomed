'''
TrainingPlan definition for torchnn ML framework
'''

from typing import Any, Dict, Union, Callable, Optional
from copy import deepcopy

import torch
import torch.nn as nn

from fedbiomed.common.constants import TrainingPlans, ProcessTypes
from fedbiomed.common.utils import get_method_spec
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.metrics import Metrics
from ._base_training_plan import BaseTrainingPlan

from opacus import PrivacyEngine 
from opacus.validators import ModuleValidator


class TorchTrainingPlan(BaseTrainingPlan, nn.Module):
    """Implements  TrainingPlan for torch NN framework

    An abstraction over pytorch module to run pytorch models and scripts on node side. Researcher model (resp. params)
    will be:

    1. saved  on a '*.py' (resp. '*.pt') files,
    2. uploaded on a HTTP server (network layer),
    3. then Downloaded from the HTTP server on node side,
    4. finally, read and executed on node side.


    Researcher must define/override:
    - a `training_data()` function
    - a `training_step()` function

    Researcher may have to add extra dependencies/python imports, by using `add_dependencies` method.
    """

    def __init__(self, model_args: dict = {}):
        """ Construct training plan

        Args:
            model_args (dict): model arguments. Items used in this class build time. Defaults to {} (empty
            dictionary)
        """

        super().__init__()

        self.__type = TrainingPlans.TorchTrainingPlan

        # cannot use it here !!!! FIXED in training_routine
        self.optimizer = None

        # data loading // should be moved to another class
        self.batch_size = 100
        self.shuffle = True

        # TODO : add random seed init
        # self.random_seed_params = None
        # self.random_seed_shuffling_data = None

        # device to use: cpu/gpu
        # - all operations except training only use cpu
        # - researcher doesn't request to use gpu by default
        self.device_init = "cpu"
        self.device = self.device_init
        if not isinstance(model_args, dict):
            self.use_gpu = False
        else:
            self.use_gpu = model_args.get('use_gpu', False)

        # list dependencies of the model
        self.add_dependency(["import torch",
                             "import torch.nn as nn",
                             "import torch.nn.functional as F",
                             "from fedbiomed.common.training_plans import TorchTrainingPlan",
                             "from fedbiomed.common.data import DataManager",
                             "from fedbiomed.common.constants import ProcessTypes",
                             "from torch.utils.data import DataLoader",
                             "from torchvision import datasets, transforms"
                             ])

        # Aggregated model parameters
        self.init_params = None

        # Differential Privacy support
        self.dp_args = None

    def type(self) -> TrainingPlans.TorchTrainingPlan:
        """ Gets training plan type"""
        return self.__type

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
            use_gpu = self.use_gpu
        use_cuda = cuda_available and ((use_gpu and node_args['gpu']) or node_args['gpu_only'])

        if node_args['gpu_only'] and not cuda_available:
            logger.error('Node wants to force model training on GPU, but no GPU is available')
        if use_cuda and not use_gpu:
            logger.warning('Node enforces model training on GPU, though it is not requested by researcher')
        if not use_cuda and use_gpu:
            logger.warning('Node training model on CPU, though researcher requested GPU')

        # Set device for training
        self.device = "cpu"
        if use_cuda:
            if node_args['gpu_num'] is not None:
                if node_args['gpu_num'] in range(torch.cuda.device_count()):
                    self.device = "cuda:" + str(node_args['gpu_num'])
                else:
                    logger.warning(f"Bad GPU number {node_args['gpu_num']}, using default GPU")
                    self.device = "cuda"
            else:
                self.device = "cuda"
        logger.debug(f"Using device {self.device} for training "
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

    def training_step(self):
        """All subclasses must provide a training_step the purpose of this actual code is to detect that it
        has been provided

        Raises:
             FedbiomedTrainingPlanError: if called and not inherited
        """
        msg = ErrorNumbers.FB303.value + ": training_step must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine(self,
                         epochs: int = 2,
                         log_interval: int = 10,
                         lr: Union[int, float] = 1e-3,
                         batch_maxnum: int = 0,
                         dry_run: bool = False,
                         use_gpu: Union[bool, None] = None,
                         fedprox_mu: float = None,
                         dp_args: Optional[dict] = None,
                         history_monitor: Any = None,
                         node_args: Union[dict, None] = None):
        # FIXME: add betas parameters for ADAM solver + momentum for SGD
        # FIXME 2: remove parameters specific for validation specified in the
        # training routine
        """Training routine procedure.

        End-user should define;

        - a `training_data()` function defining how sampling / handling data in node's dataset is done. It should
            return a generator able to output tuple (batch_idx, (data, targets)) that is iterable for each batch.
        - a `training_step()` function defining how cost is computed. It should output loss values for backpropagation.

        Args:
            epochs: Number of epochs (complete pass on data).
            log_interval: Frequency of logging loss values during training.
            lr: Learning rate.
            batch_maxnum: Equals number of data devided by batch_size, and taking the closest lower integer.
            dry_run: Whether to stop once the first batch size of the first epoch of the first round is completed.
            use_gpu: researcher requests to use GPU (or not) for training during this round (ie overload the object
                default use_gpu value) if available on node and proposed by node Defaults to None (don't overload the
                object default value)
            fedprox_mu: mu parameter in case of FredProx computing. Default is None, which means that
                FredProx is not triggered
            history_monitor: Monitor handler for real-time feed. Defined by the Node and can't be overwritten
            node_args: command line arguments for node. Can include:
                - `gpu (bool)`: propose use a GPU device if any is available. Default False.
                - `gpu_num (Union[int, None])`: if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available. Default None.
                - `gpu_only (bool)`: force use of a GPU device if any available, even if researcher
                    doesn't request for using a GPU. Default False.
        """

        if dp_args is not None:
            self.initialize_dp(dp_args)

        # set correct type for node args
        if not isinstance(node_args, dict):
            node_args = {}

        if self.optimizer is None:
            self.make_optimizer(lr)

        if self.dp_args:
            # Add __preprocess_ldp as preprocess 
            self.add_preprocess(method=self.__preprocess_ldp, process_type=ProcessTypes.DATA_LOADER)

        # Run preprocess when everything is ready before the training
        self.__preprocess()

        self._set_device(use_gpu, node_args)
        # send all model to device, ensures having all the requested tensors
        self.to(self.device)

        self.train()  # pytorch switch for training

        # Initialize training data that comes from Round class
        # TODO: Decide whether it should attached to `self`
        # self.data = data_loader

        # initial aggregated model parameters
        self.init_params = deepcopy(self.state_dict())

        for epoch in range(1, epochs + 1):
            # (below) sampling data (with `training_data` method defined on
            # researcher's notebook)
            # training_data = self.training_data(batch_size=batch_size)
            for batch_idx, (data, target) in enumerate(self.training_data_loader):

                # Plus one since batch_idx starts from 0
                batch_ = batch_idx + 1

                self.train()  # model training
                data, target = self.send_to_device(data, self.device), self.send_to_device(target, self.device)
                self.optimizer.zero_grad()

                res = self.training_step(data, target)  # raises an exception if not provided

                # If FedProx is enabled: use regularized loss function
                if fedprox_mu is not None:
                    try:
                        _mu = float(fedprox_mu)
                    except ValueError:
                        msg = ErrorNumbers.FB605.value + ": fedprox_mu parameter requested is not a float"
                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)

                    res += _mu / 2 * self.__norm_l2()

                res.backward()

                self.optimizer.step()

                if self.dp_args:
                    # To be used by the nodes to assess budget locally
                    eps, alpha = self.privacy_engine.accountant.get_privacy_spent(delta=.1/len(self.training_data_loader))

                # do not take into account more than batch_maxnum
                # batches from the dataset
                if batch_ % log_interval == 0:
                    batch_size = self.training_data_loader.batch_size
                    num_samples_till_now = min(batch_*batch_size, len(self.training_data_loader.dataset))
                    logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        num_samples_till_now,
                        len(self.training_data_loader.dataset),
                        100 * batch_ / len(self.training_data_loader),
                        res.item()))

                    # Send scalar values via general/feedback topic
                    if history_monitor is not None:
                        history_monitor.add_scalar(metric={'Loss': res.item()},
                                                   iteration=batch_,
                                                   epoch=epoch,
                                                   train=True,
                                                   num_batches=len(self.training_data_loader),
                                                   total_samples=len(self.training_data_loader.dataset),
                                                   batch_samples=batch_size)

                    if dry_run:
                        self.to(self.device_init)
                        torch.cuda.empty_cache()
                        return

                if (batch_maxnum > 0) and (batch_ >= batch_maxnum):
                    # print('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    logger.info('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    break

        # release gpu usage as much as possible though:
        # - it should be done by deleting the object
        # - and some gpu memory remains used until process (cuda kernel ?) finishes
        self.to(self.device_init)
        torch.cuda.empty_cache()

    def testing_routine(self,
                        metric: Union[MetricTypes, None],
                        metric_args: Dict[str, Any],
                        history_monitor: Any,
                        before_train: Union[bool, None] = None):
        """Performs validation routine on validation partition of the dataset

        Validation routine can be run any time after train and validation split is done. Method sends validation result
        back to researcher component as real-time.

        Args:
            metric: Metric that will be used for validation
            metric_args: The arguments for corresponding metric function.
                Please see [`sklearn.metrics`][sklearn.metrics]
            history_monitor: Real-time feed-back handler for validation results
            before_train: Declares whether is performed before training model or not.

        Raises:
            FedbiomedTrainingPlanError: if the training is failed by any reason

        """
        # TODO: Add preprocess option for testing_data_loader

        if self.testing_data_loader is None:
            msg = ErrorNumbers.FB605.value + ": can not find dataset for validation."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Build metrics object
        metric_controller = Metrics()
        tot_samples = len(self.testing_data_loader.dataset)

        self.eval()  # pytorch switch for model validation
        # Complete prediction over batches
        with torch.no_grad():
            # Data Loader for testing partition includes entire dataset in the first batch
            for batch_ndx, (data, target) in enumerate(self.testing_data_loader):
                batch_ = batch_ndx + 1

                # If `testing_step` is defined in the TrainingPlan
                if hasattr(self, 'testing_step'):
                    try:
                        m_value = self.testing_step(data, target)
                    except Exception as e:
                        # catch exception because we are letting the user design this
                        # `evaluation_step` method of the training plan
                        msg = ErrorNumbers.FB605.value + \
                            ": error then executing `testing_step` :" + \
                            str(e)

                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)

                    # If custom validation step returns None
                    if m_value is None:
                        msg = ErrorNumbers.FB605.value + \
                            ": metric function returned None"

                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)

                    metric_name = 'Custom'

                # Otherwise, check a default metric is defined
                # Use accuracy as default metric
                else:

                    if metric is None:
                        metric = MetricTypes.ACCURACY
                        logger.info(f"No `testing_step` method found in TrainingPlan and `test_metric` is not defined "
                                    f"in the training arguments `: using default metric {metric.name}"
                                    " for model validation")
                    else:
                        logger.info(
                            f"No `testing_step` method found in TrainingPlan: using defined metric {metric.name}"
                            " for model validation.")

                    metric_name = metric.name

                    try:
                        # Pass data through network layers
                        pred = self(data)
                    except Exception as e:
                        # Pytorch does not provide any means to catch exception (no custom Exceptions),
                        # that is why we need to trap general Exception
                        msg = ErrorNumbers.FB605.value + \
                            ": error - " + \
                            str(e)
                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)

                    # Convert prediction and actual values to numpy array
                    y_true = target.detach().numpy()
                    predicted = pred.detach().numpy()
                    m_value = metric_controller.evaluate(y_true=y_true, y_pred=predicted, metric=metric, **metric_args)

                metric_dict = self._create_metric_result_dict(m_value, metric_name=metric_name)

                logger.debug('Validation: Batch {} [{}/{}] | Metric[{}]: {}'.format(
                    str(batch_), batch_ * len(target), tot_samples, metric_name, m_value))

                # Send scalar values via general/feedback topic
                if history_monitor is not None:
                    history_monitor.add_scalar(metric=metric_dict,
                                               iteration=batch_,
                                               epoch=None,  # no epoch
                                               test=True,
                                               test_on_local_updates=False if before_train else True,
                                               test_on_global_updates=before_train,
                                               total_samples=tot_samples,
                                               batch_samples=len(target),
                                               num_batches=len(self.testing_data_loader))

        del metric_controller

    # provided by fedbiomed
    def save(self, filename: str, params: dict = None) -> None:
        """Save the torch training parameters from this training plan or from given `params` to a file

        Args:
            filename: Path to the destination file
            params: Parameters to save to a file, should be structured as a torch state_dict()

        """
        if params is not None:
            return torch.save(params, filename)
        else:
            return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename: str, to_params: bool = False) -> dict:
        """Load the torch training parameters to this training plan or to a data structure from a file

        Args:
            filename: path to the source file
            to_params: if False, load params to this pytorch object; if True load params to a data structure

        Returns:
            Contains parameters
        """
        params = torch.load(filename)
        if to_params is False:
            self.load_state_dict(params)
        return params

    def after_training_params(self) -> dict:
        """Retrieve parameters after training is done

        Call the user defined postprocess function:
            - if provided, the function is part of pytorch model defined by the researcher
            - and expect the model parameters as argument

        Returns:
            The state_dict of the model, or modified state_dict if preprocess is present
        """

        try:
            # Check whether postprocess method exists, and use it
            logger.debug("running model.postprocess() method")
            post_params = self.postprocess(self.state_dict())  # Post process
            if self.dp_args:
                return self.postprocess_dp(post_params) 
            return post_params
        except AttributeError:
            # Method does not exist; skip
            logger.debug("model.postprocess() method not provided")
            if self.dp_args:
                return self.postprocess_dp(self.state_dict())
            else: 
                pass

        return self.state_dict()

    def initialize_dp(self, dp_args: dict):
        """Initialize arguments to perform DP training, and check that the user
        has correctly provided all requested DP parameters in the correct form

        Args:
            dp_args (dict, optional): DP parameters provided by the user
        """
        assert 'type' in dp_args,  "DP 'type' not provided"
        assert 'sigma' in dp_args, "DP 'sigma' parameter not provided"
        assert 'clip' in dp_args,  "DP 'clip' parameter not provided"
        assert dp_args['type'] in ['central','local'], "DP strategy unknown"

        if dp_args['type'] == 'local':
            try:
                float(dp_args['sigma'])
            except ValueError:
                msg = ErrorNumbers.FB605.value + ": 'sigma'  parameter requested is not a float"
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)
        else:
            dp_args.update(sigma_CDP=dp_args['sigma'])
            dp_args['sigma'] = 0.

        try:
            float(dp_args['clip'])
        except ValueError:
            msg = ErrorNumbers.FB605.value + ": 'clip'  parameter requested is not a float"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        self.dp_args = dp_args

    def postprocess_dp(self, params: dict) -> dict:
        """
        Postprocess of model's parameters after training with DP.

        Postprocess of DP parameters implies:
        - If central DP is enabled, model's parameters are perturbed
             according to the provided DP parameters
        - When the Opacus `PrivacyEngine` is attached to the model, 
            parameters' names are modified by the addition of `_module.`. 
            This modification should be undone before communicating to the master
            for aggregation. This is needed in order to correctly perform 
            download/upload of model's parameters in the following rounds

        Args:
            params (dict): optimized parameters

        Returns:
            dict containing (postprocessed) parameters

        Exceptions:
            none
        """

        if self.dp_args['type'] == 'central':

            sigma_CDP = deepcopy(self.dp_args['sigma_CDP'])
            
            delta_params = {}
            perturbed_params = {}
            for name, param in params.items():
                ###
                ### Extracting the update
                ###
                delta_theta = deepcopy(param)
                delta_params[name] = delta_theta - self.init_params[name]

            for key, delta_param in delta_params.items():
                ###
                ### Perturb update and update parameters
                ###
                delta_theta_tilde = deepcopy(delta_param)
                delta_theta_tilde += sigma_CDP*self.dp_args['clip'] * torch.randn_like(delta_theta_tilde)
                perturbed_params[key]= delta_theta_tilde + self.init_params[key]
            
            params = deepcopy(perturbed_params )

        params_keys = list(params.keys())
        for key in params_keys:
            if '_module' in key:
                newkey = key.replace('_module.', '')
                params[newkey] = params.pop(key)

        return params
    
    def make_optimizer(self,lr: float):
        """
        Method to define the desired optimizer as class attribute `self.optimizer`.
        
        The default optimizer for torch NN models is `torch.optim.Adam`.
        The user can overwrite this method to provide a custom optimizer.

        Args:
            lr (float): learning rate
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def __norm_l2(self) -> float:
        """
        Used by FedProx optimization: evaluate L2 norm of model parameters

        Returns:
            norm (float): L2 norm of model parameters
        """
        norm = 0
        for key, val in self.state_dict().items():
            norm += ((val - self.init_params[key]) ** 2).sum()
        return norm

    def __preprocess(self):
        """Executes registered preprocess that are defined by user."""
        for (name, process) in self.pre_processes.items():
            method = process['method']
            process_type = process['process_type']

            if process_type == ProcessTypes.DATA_LOADER:
                self.__process_data_loader(method=method)
            else:
                logger.error(f"Process `{process_type}` is not implemented for `TorchTrainingPlan`. Preprocess will "
                             f"be ignored")

    def validate_and_fix_model(self):
        from opacus.validators import ModuleValidator
        # Validate and Fix model to be DP-compliant
        if not ModuleValidator.is_valid(self.model):
            print('######################################## Fixing Model ########################################')
            self.model = ModuleValidator.fix(self.model)


    def __preprocess_ldp(self, data_loader):
        """
        Method to enable DP training using Opacus.
        
        This method is executed before starting the training epochs loop any time
        the DP training is enabled. DP training is done here by using the `Opacus`
        `PrivacyEngine`. Further information are available in the 
        [Opacus webpage](https://opacus.ai/api/privacy_engine.html)

        Args:
            data_loader

        Returns:
            data_loader
        """

        # enter PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, data_loader = self.privacy_engine.make_private(module=self.model,
                                                                              optimizer=self.optimizer,
                                                                              data_loader = data_loader,
                                                                              noise_multiplier=self.dp_args.get('sigma'),
                                                                              max_grad_norm=self.dp_args.get('clip')
                                                                    )
        
        return data_loader


    def __process_data_loader(self, method: Callable):
        """Process handler for data loader kind processes.

        Args:
            method: Process method that is going to be executed

        Raises:
             FedbiomedTrainingPlanError: Raised if number of arguments of method is different than 1.
                    - triggered if execution of method fails
                    - triggered if type of the output of the method is not an instance of
                        `self.training_data_loader`
        """
        argspec = get_method_spec(method)
        if len(argspec) != 1:
            msg = ErrorNumbers.FB605.value + \
                ": process for type `PreprocessType.DATA_LOADER` should have only one argument/parameter"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        try:
            data_loader = method(self.training_data_loader)
        except Exception as e:
            msg = ErrorNumbers.FB605.value + \
                ": error while running process method -> `{method.__name__}` - " + \
                str(e)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Debug after running preprocess
        logger.debug(f'The process `{method.__name__}` has been successfully executed.')

        if isinstance(data_loader, type(self.training_data_loader)):
            self.training_data_loader = data_loader
            logger.debug(f'Data loader for training routine has been updated by the process `{method.__name__}` ')
        else:
            msg = ErrorNumbers.FB605.value + \
                ": the input argument of the method `preprocess` is `data_loader`" + \
                " and expected return value should be an instance of: " + \
                type(self.training_data_loader) + \
                " instead of " + \
                type(data_loader)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)