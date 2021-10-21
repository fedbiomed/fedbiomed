#
# linear inheritance of torch nn.Module
#


import inspect
from typing import Union, List

import torch
import torch.nn as nn

from fedbiomed.common.logger import logger


class TorchTrainingPlan(nn.Module):
    def __init__(self):
        """
        An abstraction over pytorch module to run
        pytroch models and scritps on node side.

        Researcher model (resp. params) will be 1) saved  on a '*.py'
        (resp. '*.pt') files, 2) uploaded on a HTTP server (network layer),
         3) then Downloaded from the HTTP server on node side,
         and 4) finally read and executed on node side.


        Researcher must define/override:
        - a `training_data()` function
        - a `training_step()` function

        Researcher may have to add extra dependancies/ python imports,
        by using `add_dependencies` method.
        """
        super(TorchTrainingPlan, self).__init__()

        # cannot use it here !!!! FIXED in training_routine
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        self.optimizer = None

        # data loading // should be moved to another class
        self.batch_size = 100
        self.shuffle = True
        # TODO : add random seed init
        # self.random_seed_params = None
        # self.random_seed_shuffling_data = None

        # training // may be changed in training_routine ??
        self.device = "cpu"

        # list dependencies of the model
        self.dependencies = ["from fedbiomed.common.torchnn import TorchTrainingPlan",
                             "import torch",
                             "import torch.nn as nn",
                             "import torch.nn.functional as F",
                             "from torch.utils.data import DataLoader",
                             "from torchvision import datasets, transforms"
                             ]

        # to be configured by setters
        self.dataset_path = None


    #################################################
    # provided by fedbiomed
    # FIXME: add betas parameters for ADAM solver + momentum for SGD
    def training_routine(self,
                         epochs: int = 2,
                         log_interval: int = 10,
                         lr: Union[int, float] = 1e-3,
                         batch_size: int = 48,
                         batch_maxnum: int = 0,
                         dry_run: bool = False,
                         monitor=None):
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
            epochs (int, optional): number of epochs (complete pass on data).
            Defaults to 2.
            log_interval (int, optional): frequency of logging. Defaults to 10.
            lr (Union[int, float], optional): learning rate. Defaults to 1e-3.
            batch_size (int, optional): size of batch. Defaults to 48.
            batch_maxnum (int, optional): equals number of data devided
            by batch_size,
            and taking the closest lower integer. Defaults to 0.
            dry_run (bool, optional): whether to stop once the first
            batch size of the first epoch of the first round is completed.
            Defaults to False.
            monitor ([type], optional): [description]. Defaults to None.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")
        self.device = "cpu"


        for epoch in range(1, epochs + 1):
            # (below) sampling data (with `training_data` method defined on
            # researcher's notebook)
            training_data = self.training_data(batch_size=batch_size)
            for batch_idx, (data, target) in enumerate(training_data):
                self.train()  # model training
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # (below) calling method `training_step` defined on
                # researcher's notebook
                res = self.training_step(data, target)
                res.backward()
                self.optimizer.step()

                # do not take into account more than batch_maxnum
                # batches from the dataset
                if (batch_maxnum > 0) and (batch_idx >= batch_maxnum):
                    #print('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    logger.debug('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    break

                if batch_idx % log_interval == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(training_data.dataset),
                        100 * batch_idx / len(training_data),
                        res.item()))

                    # Send scalar values via general/feedback topic
                    if monitor is not None:
                        monitor.add_scalar('Loss', res.item(), batch_idx, epoch)

                    if dry_run:
                        return

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep: List[str]):
        """adds extra python import(s)

        Args:
            dep (List[str]): package name import, eg: 'import torch as th'
        """
        self.dependencies.extend(dep)
        pass

    # provided by fedbiomed
    def save_code(self, filename: str):
        """Save the class code for this training plan to a file

        Args:
            filename (string): path to the destination file

        Returns:
            None

        Exceptions:
            none
        """

        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)
        logger.debug("torchnn saved model filename: " + filename)
        # TODO: try/except
        file = open(filename, "w")
        # (above) should we write it in binary (for the sake of space
        # optimization)?
        file.write(content)
        file.close()

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
            return(torch.save(params, filename))
        else:
            return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename: str, to_params: bool = False) -> dict:
        """Load the torch training parameters to this training plan or
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

    # provided by the fedbiomed // should be moved in a DATA
    # manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        logger.debug('Dataset_path' + self.dataset_path)

    # provided by the fedbiomed // should be moved in a DATA
    # manipulation module
    def training_data(self, batch_size=48):
        """
        A method that describes how to parse/select/shuffle data
        when training model. Should be defined by researcher in its
        trainig plan.

        Args:
            batch_size (int, optional): size of the batch. Defaults to 48.
        """

        pass

    def after_training_params(self):
        return self.state_dict()
