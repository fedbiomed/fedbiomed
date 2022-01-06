#!/usr/bin/env python
# coding: utf-8

# # Fedbiomed Researcher

# Use for developing (autoreloads changes made across packages)

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
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node add`
# 3. Run the node using `./scripts/fedbiomed_run node add`. Wait until you get `Starting task manager`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch.nn MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.torchnn import TorchTrainingPlan
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# you can use any class name eg:
# class AlterTrainingPlan(TorchTrainingPlan):
class MyTrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super(MyTrainingPlan, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # Here we define the custom dependencies that will be needed by our custom Dataloader
        # In this case, we need the torch DataLoader classes
        # Since we will train on MNIST, we need datasets and transform from torchvision
        deps = ["from torchvision import datasets, transforms",
               "from torch.utils.data import DataLoader"]
        self.add_dependency(deps)

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
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        print("[INFO] Training on dataset: " + str(self.dataset_path))
        dataset1 = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        return data_loader

    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss


# This group of arguments correspond respectively:
# * `model_args`: a dictionary with the arguments related to the model (e.g. number of layers, features, etc.). This will be passed to the model class on the node side.
# * `training_args`: a dictionary containing the arguments for the training routine (e.g. batch size, learning rate, epochs, etc.). This will be passed to the routine on the node side.
#
# **NOTE:** typos and/or lack of positional (required) arguments will raise error. 🤓

model_args = {}

training_args = {
    'batch_size': 48,
    'lr': 1e-3,
    'epochs': 1,
    'dry_run': False,
    'batch_maxnum': 100  # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}


#    Define an experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `rounds` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                #nodes=None,
                model_class=MyTrainingPlan,
                # model_class=AlterTrainingPlan,
                # model_path='/path/to/model_file.py',
                model_args=model_args,
                training_args=training_args,
                rounds=rounds,
                aggregator=FedAverage(),
                node_selection_strategy=None,
                save_breakpoints=True)


# Let's start the experiment.
# By default, this function doesn't stop until all the `rounds` are done for all the nodes
# You can interrupt the exp.run() after one round,
# and then reload the breakpoint and continue the training.

exp.run()


# Local training results for each round and each node are available in `exp.training_replies` (index 0 to (`rounds` - 1) ).
# For example you can view the training results for the last round below.
#
# Different timings (in seconds) are reported for each dataset of a node participating in a round :
# - `rtime_training` real time (clock time) spent in the training function on the node
# - 'ptime_training` process time (user and system CPU) spent in the training function on the node
# - `rtime_total` real time (clock time) spent in the researcher between sending the request and handling the response, at the `Job()` layer

print("______________ original training replies_________________")
print("\nList the training rounds : ", exp.training_replies.keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = exp.training_replies[rounds - 1].data
for c in range(len(round_data)):
    print("\t- {id} :\
        \n\t\trtime_training={rtraining:.2f} seconds\
        \n\t\tptime_training={ptraining:.2f} seconds\
        \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['node_id'],
                rtraining = round_data[c]['timing']['rtime_training'],
                ptraining = round_data[c]['timing']['ptime_training'],
                rtotal = round_data[c]['timing']['rtime_total']))
print('\n')


# ## Delete experiment

del exp
# here we simulate the removing of the ongoing experiment
# fret not! we have saved breakpoint, so we can retrieve parameters
# of the experiment using `load_breakpoint` method


# ## Resume an experiment
# 
# While experiment is running, you can shut it down (after the first round) and resume the experiment from the next cell. Or wait for the experiment completion.
# 
# 
# **To load the latest breakpoint of the latest experiment**
# 
# Run :
# `Experiment.load_breakpoint()`. It reloads latest breakpoint, and will bypass `search` method
# 
# and then use `.run` method as you would do with an existing experiment.
# 
# **To load a specific breakpoint** specify breakpoint folder when using `Experiment.load_breakpoint("./var/breakpoints/Experiment_xx/breakpoint_yy)`. Replace `xx` and `yy` by the real values.
# 

loaded_exp = Experiment.load_breakpoint()

print(f'Experimention folder: {loaded_exp.experimentation_folder}')
print(f'Loaded experiment path: {loaded_exp.experimentation_path}')

# Continue training for the experiment loaded from breakpoint.
# If you ran all the rounds and load the last breakpoint, there won't be any more round to run.

loaded_exp.run()


print("______________ loaded training replies_________________")
print("\nList the training rounds : ", loaded_exp.training_replies.keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = loaded_exp.training_replies[rounds - 1].data
for c in range(len(round_data)):
    #print(round_data[c])
    print("\t- {id} :\
        \n\t\trtime_training={rtraining:.2f} seconds\
        \n\t\tptime_training={ptraining:.2f} seconds\
        \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['node_id'],
                rtraining = round_data[c]['timing']['rtime_training'],
                ptraining = round_data[c]['timing']['ptime_training'],
                rtotal = round_data[c]['timing']['rtime_total']))
print('\n')

print(loaded_exp.training_replies[rounds - 1].dataframe)


# Federated parameters for each round are available in `exp.aggregated_params` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", loaded_exp.aggregated_params.keys())

print("\nAccess the federated params for training rounds : ")
for round in loaded_exp.aggregated_params.keys():
  print("round {r}".format(r=round))
  print("\t- params_path: ", loaded_exp.aggregated_params[round]['params_path'])
  print("\t- parameter data: ", loaded_exp.aggregated_params[round]['params'].keys())


# ## Optional : searching the data

#from fedbiomed.researcher.requests import Requests
#
#r = Requests()
#data = r.search(tags)
#
#import pandas as pd
#for node_id in data.keys():
#    print('\n','Data for ', node_id, '\n\n', pd.DataFrame(data[node_id]))
#
#print('\n')
