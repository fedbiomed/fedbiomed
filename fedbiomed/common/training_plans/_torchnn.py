'''
TrainingPlan definition for torchnn ML framework
'''

import inspect
from typing import Union, List
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fedbiomed.common.constants import TrainingPlans

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from ._base_training_plan import BaseTrainingPlan


class TorchTrainingPlan(BaseTrainingPlan, nn.Module):
    """
    Implementation of TraingPlan for torch NN framework
    """

    def __init__(self, model_args: dict = {}):
        """
        An abstraction over pytorch module to run
        pytorch models and scripts on node side.

        Researcher model (resp. params) will be 1) saved  on a '*.py'
        (resp. '*.pt') files, 2) uploaded on a HTTP server (network layer),
         3) then Downloaded from the HTTP server on node side,
         and 4) finally read and executed on node side.


        Researcher must define/override:
        - a `training_data()` function
        - a `training_step()` function

        Researcher may have to add extra dependancies/ python imports,
        by using `add_dependencies` method.

        Args:
          - model_args (dict, optional): model arguments. Items used in this class:
            - use_gpu (bool, optional) : researcher requests to use GPU (or not) for training
                if available on node and proposed by node. Defaults to False.
        """

        super(TorchTrainingPlan, self).__init__()

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

        self.add_dependency(["from fedbiomed.common.training_plans import TorchTrainingPlan",
                             "from fedbiomed.common.data import DataManager",
                             "import torch",
                             "import torch.nn as nn",
                             "import torch.nn.functional as F",
                             "from torch.utils.data import DataLoader",
                             "from torchvision import datasets, transforms"
                             ])

        # Aggregated model parameters
        self.init_params = None

    def type(self):
        """ Getter for training plan type """

        return self.__type

    def _set_device(self, use_gpu: Union[bool, None], node_args: dict):
        """
        Set device (CPU, GPU) that will be used for training, based on `node_args`

        Args:
          - use_gpu (Union[bool, None]) : researcher requests to use GPU (or not)
          - node_args (dict): command line arguments for node
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

    def training_step(self):
        """
        all subclasses must provide a training_steproutine
        the purpose of this actual code is to detect that it has been provided

        :raise FedbiomedTrainingPlanError if called
        """
        msg = ErrorNumbers.FB303.value + ": training_step must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def training_routine(self,
                         training_data: DataLoader,
                         epochs: int = 2,
                         log_interval: int = 10,
                         lr: Union[int, float] = 1e-3,
                         batch_size: int = 48,
                         batch_maxnum: int = 0,
                         dry_run: bool = False,
                         use_gpu: Union[bool, None] = None,
                         fedprox_mu=None,
                         monitor=None,
                         node_args: Union[dict, None] = None):
        # FIXME: add betas parameters for ADAM solver + momentum for SGD
        """
        Training routine procedure.

        Researcher should defined :
        - a `training_data()` function defining
        how sampling / handling data in node's dataset is done.
        It should return a generator able to ouput tuple
        (batch_idx, (data, targets)) that is iterable for each batch.
        - a `training_step()` function defining how cost is computed. It should
        output model error for model backpropagation.

        Args:
            - epochs (int, optional): number of epochs (complete pass on data).
                Defaults to 2.
            - log_interval (int, optional): frequency of logging. Defaults to 10.
                lr (Union[int, float], optional): learning rate. Defaults to 1e-3.
            - batch_size (int, optional): size of batch. Defaults to 48.
            - batch_maxnum (int, optional): equals number of data devided
                by batch_size,
                and taking the closest lower integer. Defaults to 0.
            - dry_run (bool, optional): whether to stop once the first
            - batch size of the first epoch of the first round is completed.
                Defaults to False.
            - use_gpu (Union[bool, None], optional) : researcher requests to use GPU (or not)
                for training during this round (ie overload the object default use_gpu value)
                if available on node and proposed by node
                Defaults to None (dont overload the object default value)
            - fedprox_mu (float or None): mu parameter in case of FredProx computing. Fedault is None, which means that FredProx is not triggered
            - monitor ([type], optional): [description]. Defaults to None.
            - node_args (Union[dict, None]): command line arguments for node. Can include:
                - gpu (bool): propose use a GPU device if any is available. Default False.
                - gpu_num (Union[int, None]): if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available. Default None.
                - gpu_only (bool): force use of a GPU device if any available, even if researcher
                    doesnt request for using a GPU. Default False.
        """
        # set correct type for node args
        if not isinstance(node_args, dict):
            node_args = {}

        self._set_device(use_gpu, node_args)
        # send all model to device, ensures having all the requested tensors
        self.to(self.device)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize training data that comes from Round class
        self.data = training_data

        # initial aggregated model parameters
        self.init_params = deepcopy(self.state_dict())

        for epoch in range(1, epochs + 1):
            # (below) sampling data (with `training_data` method defined on
            # researcher's notebook)
            # training_data = self.training_data(batch_size=batch_size)
            for batch_idx, (data, target) in enumerate(self.data):
                self.train()  # model training
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                res = self.training_step(data, target)  # raises an exception if not provided

                # If FedProx is enabled: use regularized loss function
                if fedprox_mu is not None:
                    try:
                        _mu = float(fedprox_mu)
                    except ValueError:
                        msg = ErrorNumbers.FB605.value + ": fedprox_mu parameter reuqested nut is not a float"
                        logger.critical(msg)
                        raise FedbiomedTrainingPlanError(msg)

                    res += _mu / 2 * self.__norm_l2()

                res.backward()

                self.optimizer.step()

                # do not take into account more than batch_maxnum
                # batches from the dataset
                if (batch_maxnum > 0) and (batch_idx >= batch_maxnum):
                    # print('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    logger.debug('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    break

                if batch_idx % log_interval == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(self.data.dataset),
                        100 * batch_idx / len(self.data),
                        res.item()))

                    # Send scalar values via general/feedback topic
                    if monitor is not None:
                        monitor.add_scalar('Loss', res.item(), batch_idx, epoch)

                    if dry_run:
                        self.to(self.device_init)
                        torch.cuda.empty_cache()
                        return

        # release gpu usage as much as possible though:
        # - it should be done by deleting the object
        # - and some gpu memory remains used until process (cuda kernel ?) finishes
        self.to(self.device_init)
        torch.cuda.empty_cache()

    # provided by fedbiomed
    def save(self, filename, params: dict = None) -> None:
        """Save the torch training parameters from this training plan or
        from given `params` to a file

        Args:
            filename (string): path to the destination file
            params (dict, optional): parameters to save to a file, should
            be structured as a torch state_dict()

        Returns:
            pytorch.save() returns None

        Exceptions:
            none
        """
        if params is not None:
            return (torch.save(params, filename))
        else:
            return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename: str, to_params: bool = False) -> dict:
        """
        Load the torch training parameters to this training plan or
        to a data structure from a file

        Args:
            filename (string): path to the source file
            to_params (bool, optional): if False, load params to this
            pytorch object;
            if True load params to a data structure

        Returns:
            dict containing parameters

        Exceptions:
            none
        """

        params = torch.load(filename)
        if to_params is False:
            self.load_state_dict(params)
        return params

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval=10):
        pass

    def after_training_params(self):
        '''
        effectively call the user defined postprocess function (if provided)

        - if provided, the function ispart of pytorch model defined by the researcher
        - and expect the model parameters as argument

        and returns the (modified) state_dict of the model
        '''
        try:
            # Check whether postprocess method exists, and use it
            logger.debug("running model.postprocess() method")
            return self.postprocess(self.state_dict())
        except AttributeError:
            # Method does not exist; skip
            logger.debug("model.postprocess() method not provided")
            pass

        return self.state_dict()

    def __norm_l2(self):
        """
        used by FedProx optimization
        """
        norm = 0
        for key, val in self.state_dict().items():
            norm += ((val - self.init_params[key]) ** 2).sum()
        return norm
