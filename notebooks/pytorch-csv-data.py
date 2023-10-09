#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher to train a model on a CSV dataset

# This example shows how to use a CSV format file as a node dataset. The example CSV
# file is synthetic data with a format inspired from ADNI dataset.

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
## Start the network and setting the node up
# Before running this notebook, you should start the network from fedbiomed-network
# Therefore, it is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 1 to add a csv file to the node
#   * Choose the name, tags and description of the dataset
#     * use `#test_data`` for the tags
#   * Pick the .csv file from your PC (here: pseudo_adni_mod.csv)
#   * Data must have been added
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node list`
# 3. Run the node using `./scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch training plan MyTrainingPlan class to send for training on the node
import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
import pandas as pd

# Here we define the model to be used. 
# You can use any class name (here 'MyTrainingPlan')
class MyTrainingPlan(TorchTrainingPlan):
    
    # Model 
    def init_model(self, model_args):    
        model = self.Net(model_args)
        return model 
    
    
    # Dependencies
    def init_dependencies(self):
        # Here we define the custom dependencies that will be needed by our custom Dataloader
        # In this case, we need the torch Dataset and DataLoader classes
        # We need pandas to read the local .csv file at the node side
        deps = ["from torch.utils.data import Dataset",
                "import pandas as pd"]
        
        return deps
    
    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            # should match the model arguments dict passed below to the experiment class
            self.fc1 = nn.Linear(model_args['in_features'], 5)
            self.fc2 = nn.Linear(5, model_args['out_features'])

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    def training_step(self, data, target):
        output = self.model().forward(data).float()
        criterion = torch.nn.MSELoss()
        loss   = criterion(output, target.unsqueeze(1))
        return loss

    def training_data(self,  batch_size = 48):
        df = pd.read_csv(self.dataset_path, sep=';', index_col=False)
        x_dim = self.model_args()['in_features']
        x_train = df.iloc[:,:x_dim].values
        y_train = df.iloc[:,-1].values
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        
        data_manager = DataManager(dataset=x_train , target=y_train, **train_kwargs)
        
        return data_manager

# model parameters 
model_args = {
    'in_features': 15, 
    'out_features': 1
}

# training parameters 
training_args = {
    'batch_size': 20, 
    'optimizer_args': {
        'lr': 1e-3
    }, 
    'epochs': 10, 
    'dry_run': False,  
}


# Define an experiment
# - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
# - run a round of local training on nodes with model defined in `model_path` + federation with `aggregator`
# - run for `round_limit` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.federated_workflows.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

# Calling the training data with specified tags
tags =  ['#test_data']
rounds = 5

exp = Experiment(tags=tags,
                 training_plan_class=MyTrainingPlan,
                 model_args=model_args,
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

exp.training_replies()[rounds - 1].dataframe()


# Federated parameters for each round are available via `exp.aggregated_params()` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", exp.aggregated_params().keys())

print("\nAccess the federated params for the last training round : ")
print("\t- params_path: ", exp.aggregated_params()[rounds - 1]['params_path'])
print("\t- parameter data: ", exp.aggregated_params()[rounds - 1]['params'].keys())
