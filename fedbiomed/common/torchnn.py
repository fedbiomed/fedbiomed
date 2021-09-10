#
# linear inheritance of torch nn.Module
#


import inspect

import torch
import torch.nn as nn

from fedbiomed.common.logger import _FedLogger

class TorchTrainingPlan(nn.Module):
    def __init__(self):
        super(TorchTrainingPlan, self).__init__()

        # cannot use it here !!!! FIXED in training_routine
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        self.optimizer = None

        # data loading // should ne moved to another class
        self.batch_size = 100
        self.shuffle    = True

        # training // may be changed in training_routine ??
        self.device = "cpu"

        # list dependencies of the model
        self.dependencies = [ "from fedbiomed.common.torchnn import TorchTrainingPlan",
                              "import torch",
                              "import torch.nn as nn",
                              "import torch.nn.functional as F",
                              "from torch.utils.data import DataLoader",
                              "from torchvision import datasets, transforms"
                             ]

        # to be configured by setters
        self.dataset_path = None

        # get the logger from the _FedLogger class (thanks Mr Singleton)
        self.system_logger = _FedLogger()


    #################################################
    # provided by fedbiomed
    def training_routine(self, epochs=2, log_interval = 10, lr=1e-3, batch_size=48, batch_maxnum=0, dry_run=False, logger=None):

        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        self.device = "cpu"


        for epoch in range(1, epochs + 1):
            training_data = self.training_data(batch_size=batch_size)
            for batch_idx, (data, target) in enumerate(training_data):
                self.train()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                res = self.training_step(data, target)
                res.backward()
                self.optimizer.step()

                # do not take into account more than batch_maxnum batches from the dataset
                if (batch_maxnum > 0) and (batch_idx >= batch_maxnum):
                    #print('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    self.system_logger.debug('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    break

                if batch_idx % log_interval == 0:
#                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.system_logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(training_data.dataset),
                        100 * batch_idx / len(training_data),
                        res.item()))
                    #
                    # deal with the logger here
                    #

                    if dry_run:
                        return

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
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

        # try/except todo
        file = open(filename, "w")
        file.write(content)
        file.close()

    # provided by fedbiomed
    def save(self, filename, params: dict=None):
        """Save the torch training parameters from this training plan or from given `params` to a file

        Args:
            filename (string): path to the destination file
            params (dict, optional): parameters to save to a file, should be structured as a torch state_dict()

        Returns:
            torch.save() return code (documentation ?), probably None

        Exceptions:
            none
        """
        if params is not None:
            return(torch.save(params, filename))
        else:
            return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename, to_params: bool=False):
        """Load the torch training parameters to this training plan or to a data structure from a file

        Args:
            filename (string): path to the source file
            to_params (bool, optional): if False, load params to this object; if True load params to a data structure

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
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        self.system_logger.debug('Dataset_path' + self.dataset_path)

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def training_data(self, batch_size = 48):

        pass
