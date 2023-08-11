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

# Declare a torch training plan MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from torchvision import datasets, transforms


# Here we define the model to be used. 
# You can use any class name (here 'Net')
class MyTrainingPlan(TorchTrainingPlan):
    
    # Defines and return model 
    def init_model(self, model_args):
        return self.Net(model_args = model_args)
    
    # Defines and return optimizer
    def init_optimizer(self, optimizer_args):
        return torch.optim.Adam(self.model().parameters(), lr = optimizer_args["lr"])
    
    # Declares and return dependencies
    def init_dependencies(self):
        deps = ["from torchvision import datasets, transforms"]
        return deps
    
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)


            output = F.log_softmax(x, dim=1)
            return output

    def training_data(self):
        # Custom torch Dataloader for MNIST data
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        train_kwargs = { 'shuffle': True}
        return DataManager(dataset=dataset1, **train_kwargs)
    
    def training_step(self, data, target):
        output = self.model().forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss



# This group of arguments correspond respectively:
# * `model_args`: a dictionary with the arguments related to the model (e.g. number of layers, features, etc.). This will be passed to the model class on the node side.
# * `training_args`: a dictionary containing the arguments for the training routine (e.g. batch size, learning rate, epochs, etc.). This will be passed to the routine on the node side.
#
# **NOTE:** typos and/or lack of positional (required) arguments will raise error. 🤓

model_args = {}

training_args = {
    'loader_args': { 'batch_size': 48, }, 
    'optimizer_args': {
        "lr" : 1e-3
    },
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}

#    ## Declare and run the experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `rounds` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                 model_args=model_args,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)


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
print("\t- parameter data: ", exp.aggregated_params()[rounds - 1]['params'].keys())

# Feel free to run other sample notebooks or try your own models :D

