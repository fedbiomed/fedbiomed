#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher

# Use for developing (autoreloads changes made across packages)


# ## Setting the node up
# It is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 2 (default) to add MNIST to the node
#   * Confirm default tags by hitting "y" and ENTER
#   * Pick the folder where MNIST is downloaded (this is due torch issue https://github.com/pytorch/vision/issues/3549)
#   * Data must have been added (if you get a warning saying that data must be unique is because it's been already added)
#
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node add`
# 3. Run the node using `./scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch training plan MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from torchvision import datasets, transforms

# Here we define the training plan. 
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

# Define an experiment with saved breakpoints
# - search nodes serving data for these `tags`, optionally filter
#   on a list of node ID with `nodes`
# - run a round of local training on nodes with model defined in
#   `model_path` + federation with `aggregator`
# - run for `round_limit` rounds, applying the `node_selection_strategy` between the rounds
# - specify `save_breakpoints` for saving breakpoint at the end of each round.
#
# Let's call ${FEDBIOMED_DIR} the base directory where you cloned Fed-BioMed.
# Breakpoints will be saved under `Experiment_xxxx` folder at
# `${FEDBIOMED_DIR}/var/experiments/Experiment_xxxx/breakpoints_yyyy` (by default).

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
                 node_selection_strategy=None,
                 save_breakpoints=True)


# Let's start the experiment.
# By default, this function doesn't stop until all the `round_limit` rounds are done for all the nodes
# You can interrupt the exp.run() after one round,
# and then reload the breakpoint and continue the training.

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
round_data = exp.training_replies()[rounds - 1]
for r in round_data.values():
    print("\t- {id} :\
    \n\t\trtime_training={rtraining:.2f} seconds\
    \n\t\tptime_training={ptraining:.2f} seconds\
    \n\t\trtime_total={rtotal:.2f} seconds".format(id = r['node_id'],
        rtraining = r['timing']['rtime_training'],
        ptraining = r['timing']['ptime_training'],
        rtotal = r['timing']['rtime_total']))
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
# **To load a specific breakpoint** specify breakpoint folder.
# 
# - absolute path: use
#   `Experiment.load_breakpoint("${FEDBIOMED_DIR}/var/experiments/Experiment_xxxx/breakpoint_yyyy)`.
#    Replace `xxxx` and `yyyy` by the real values.
# - relative path from a notebook: a notebook is running from the `${FEDBIOMED_DIR}/notebooks` directory
#   so use `Experiment.load_breakpoint("../var/experiments/Experiment_xxxx/breakpoint_yyyy)`.
#   Replace `xxxx` and `yyyy` by the real values.
# - relative path from a script: if launching the script from the
#   ${FEDBIOMED_DIR} directory (eg: `python ./notebooks/general-breakpoint-save-resume.py`)
#  then use a path relative to the current directory eg:
# `Experiment.load_breakpoint("./var/experiments/Experiment_xxxx/breakpoint_yyyy)`


loaded_exp = Experiment.load_breakpoint()

print(f'Experimentation folder: {loaded_exp.experimentation_folder()}')
print(f'Loaded experiment path: {loaded_exp.experimentation_path()}')

# Continue training for the experiment loaded from breakpoint.
# If you ran all the rounds and load the last breakpoint, there won't be any more round to run.

loaded_exp.run()


print("\nList the training rounds : ", exp.training_replies().keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = loaded_exp.training_replies()[rounds - 1]
for r in round_data.values():
    print("\t- {id} :\
    \n\t\trtime_training={rtraining:.2f} seconds\
    \n\t\tptime_training={ptraining:.2f} seconds\
    \n\t\trtime_total={rtotal:.2f} seconds".format(id = r['node_id'],
        rtraining = r['timing']['rtime_training'],
        ptraining = r['timing']['ptime_training'],
        rtotal = r['timing']['rtime_total']))
print('\n')


# Federated parameters for each round are available via `exp.aggregated_params()` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", loaded_exp.aggregated_params().keys())

print("\nAccess the federated params for training rounds : ")
for round in loaded_exp.aggregated_params().keys():
  print("round {r}".format(r=round))
  print("\t- params_path: ", loaded_exp.aggregated_params()[round]['params_path'])
  print("\t- parameter data: ", loaded_exp.aggregated_params()[round]['params'].keys())


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
