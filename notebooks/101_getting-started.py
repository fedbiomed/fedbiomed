#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher base example

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
# ## Setting the node up
# It is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 2 (default) to add MNIST to the node
#   * Confirm default tags by hitting "y" and ENTER
#   * Pick the folder where MNIST is downloaded (this is due torch issue https://github.com/pytorch/vision/issues/3549)
#   * Data must have been added (if you get a warning saying that data must be unique is because it's been already added)
#
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node list`
# 3. Run the node using `./scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.


# ## Define an experiment model and parameters

# Declare a TorchTrainingPlan subclass to send for training on the node

from typing import Any, Dict

import torch
from torchvision import datasets, transforms

from fedbiomed.common.data import DataManager
from fedbiomed.common.training_plans import TorchTrainingPlan


class MnistTorchTrainingPlan(TorchTrainingPlan):
    """Custom torch training plan, implementing MNIST dataset loading.

    This class also overrides model-creation behaviour and training
    plan saving behavior to lighten dump files, at the cost of being
    way more verbose than its `MnistTorchTrainingPlan` counterpart.
    """

    def __init__(self) -> None:
        """Instantiate the training plan."""
        super().__init__()
        # Record all dependencies of this class's source code.
        self.add_dependency([
            "from typing import Any, Dict",
            "import torch",
            "from torchvision import datasets, transforms",
            "from fedbiomed.common.data import DataManager",
            "from fedbiomed.common.training_plans import TorchTrainingPlan",
        ])

    @staticmethod
    def init_model(model_args: Dict[str, Any]) -> torch.nn.Module:
        """Set up the neural network that needs training."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax(),
        )

    @staticmethod
    def init_loss() -> torch.nn.Module:
        """Set up the model's loss-computing module."""
        return torch.nn.NLLLoss()

    @staticmethod
    def init_optim(optim_args: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the optimizer's configuration.

        Use the input configuration, or a default Adam optimizer
        if no parameters are actually input.
        """
        if not optim_args:
            return {"lrate": 0.001, "modules": ["adam"]}
        return optim_args

    def training_data(
            self,
            dataset_path: str,
            batch_size: int = 48
        ) -> DataManager:
        """Return a DataManager wrapping the MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(
            dataset_path, train=True, download=False, transform=transform
        )
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset=dataset, **train_kwargs)



# This group of arguments correspond respectively:
# * `model_args`: a dictionary with the arguments related to the model (e.g. number of layers, features, etc.). This will be passed to the model class on the node side.
# * `training_args`: a dictionary containing the arguments for the training routine (e.g. batch size, learning rate, epochs, etc.). This will be passed to the routine on the node side.
#
# **NOTE:** typos and/or lack of positional (required) arguments will raise error. 🤓

model_args = {}

training_args = {
    'batch_size': 48,
    'epochs': 1,
    'num_updates': 100, # Fast pass for development: only use 100 steps per round
    'test_on_global_updates': True,  # test the updated model on round start
    'test_ratio': 0.2,  # use 20% of the dataset as validation
}

#    ## Declare and run the experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `rounds` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

# Use Momentum on the researcher side.
res_opt = Optimizer(lrate=1., modules=[MomentumModule()])

exp = Experiment(
    tags=tags,
    model_args=model_args,
    training_plan=MnistTorchTrainingPlan(),
    training_args=training_args,
    round_limit=rounds,
    aggregator=FedAverage(res_opt),
    node_selection_strategy=None
)


# Let's start the experiment.
# By default, this function doesn't stop until all the `round_limit` rounds are done for all the nodes

exp.run()

# Local training results for each round and each node are available via `exp.training_replies()` (index 0 to (`rounds` - 1) ).
# For example you can view the training results for the last round below.
#
# Different timings (in seconds) are reported for each dataset of a node participating in a round :
# - `rtime_training` real time (clock time) spent in the training function on the node
# - 'ptime_training` process time (user and system CPU) spent in the training function on the node
# - `rtime_total` real time (clock time) spent in the researcher between sending the request and handling the response, at the `Job()` layer

print("\nList the training rounds : ", exp.training_replies().keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = exp.training_replies()[rounds - 1].data()
for c in range(len(round_data)):
    print("\t- {id} :\
        \n\t\trtime_training={rtraining:.2f} seconds\
        \n\t\tptime_training={ptraining:.2f} seconds\
        \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['node_id'],
                rtraining = round_data[c]['timing']['rtime_training'],
                ptraining = round_data[c]['timing']['ptime_training'],
                rtotal = round_data[c]['timing']['rtime_total']))
print('\n')

print(exp.training_replies()[rounds - 1].dataframe())


# Federated parameters for each round are available via `exp.aggregated_params()` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", exp.aggregated_params().keys())

print("\nAccess the federated params for the last training round : ")
print("\t- params_path: ", exp.aggregated_params()[rounds - 1]['params_path'])
print("\t- parameter spec: ", exp.aggregated_params()[rounds - 1]['params'])

# Feel free to run other sample notebooks or try your own models :D
