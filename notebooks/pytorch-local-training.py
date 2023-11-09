#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher compare with local training

# This example shows how to run a Fed-BioMed job locally on the server, and then compare the training results with the federated training.

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`

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

# ## Create an experiment to train a model on the data found

# Declare a torch training plan MyTrainingPlan class to send for training on the node

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def training_data(self, batch_size = 48):
        # Custom torch Dataloader for MNIST data
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
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

training_args = {
    'loader_args': { 'batch_size': 48, },
    'optimizer_args': {
        'lr': 1e-3
    },
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 200 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}


# # Train the federated model

# Define an experiment
# - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
# - run a round of local training on nodes with model defined in `model_path` + federation with `aggregator`
# - run for `round_limit` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                 training_plan_class=MyTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)

# Let's start the experiment.
# 
# By default, this function doesn't stop until all the `round_limit` rounds are done for all the nodes

exp.run()


# Retrieve the federated model parameters

fed_model = exp.training_plan().model()
fed_model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
print(fed_model)


# # Local model

# Here we load the local MNIST dataset

from torchvision import datasets, transforms
from fedbiomed.researcher.environ import environ

local_mnist = os.path.join(environ['TMP_DIR'], 'local_mnist')
print(f'Using directory {local_mnist} for MNIST local copy')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

datasets.MNIST(root = local_mnist, download = True, train = True, transform = transform)


# We create an object localJob, which mimics the functionalities of the class Job to run the model on the input local dataset

# The class local job mimics the class job used in the experiment
from fedbiomed.researcher.job import localJob
from fedbiomed.researcher.environ import environ

rounds = 2

# local train on same amount of data as federated with 1 node
training_args['epochs'] *= rounds

local_job = localJob( dataset_path = local_mnist,
          training_plan_class=MyTrainingPlan,
          training_args=training_args)

# Running the localJob

local_job.start_training()


# We retrieve the local models parameters
local_model = local_job.model


# # Comparison

# We define a little testing routine to extract the accuracy metrics on the testing dataset

import torch
import torch.nn.functional as F


def testing_Accuracy(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    device = 'cpu'

    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        pred = output.argmax(dim=1, keepdim=True)

    test_loss /= len(data_loader.dataset)
    accuracy = 100* correct/len(data_loader.dataset)

    return(test_loss, accuracy)


# Loading the testing dataset and computing accuracy metrics for local and federated models

test_set = datasets.MNIST(root = local_mnist, download = True, train = False, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

acc_local = testing_Accuracy(local_model, test_loader)
acc_federated = testing_Accuracy(fed_model, test_loader)


print('\nAccuracy local training: {:.4f}, \nAccuracy federated training:  {:.4f}\nDifference: {:.4f}'.format(
             acc_local[1], acc_federated[1], acc_local[1]-acc_federated[1]))

print('\nError local training: {:.4f}, \nError federated training:  {:.4f}\nDifference: {:.4f}'.format(
             acc_local[0], acc_federated[0], acc_local[0]-acc_federated[0]))


