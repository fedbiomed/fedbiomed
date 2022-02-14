#!/usr/bin/env python
# coding: utf-8

# # Fedbiomed Researcher to train a model on a CSV dataset

# This example shows how to use a CSV format file as a node dataset. The example CSV
# file is synthetic data with a format inspired from ADNI dataset.

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
## Start the network and setting the node up
# Before running this notebook, you shoud start the network from fedbiomed-network, as detailed in https://gitlab.inria.fr/fedbiomed/fedbiomed-network
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



# Declare a torch.nn MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.torchnn import TorchTrainingPlan

from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Here we define the model to be used.
# You can use any class name (here 'MyTrainingPlan')
class MyTrainingPlan(TorchTrainingPlan):
    def __init__(self, model_args: dict = {}):
        super(MyTrainingPlan, self).__init__(model_args)
        # should match the model arguments dict passed below to the experiment class
        self.in_features = model_args['in_features']
        self.out_features = model_args['out_features']
        self.fc1 = nn.Linear(self.in_features, 5)
        self.fc2 = nn.Linear(5, self.out_features)

        # Here we define the custom dependencies that will be needed by our custom Dataloader
        # In this case, we need the torch Dataset and DataLoader classes
        # We need pandas to read the local .csv file at the node side
        deps = ["from torch.utils.data import Dataset, DataLoader",
                "import pandas as pd"]
        self.add_dependency(deps)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, data, target):
        output = self.forward(data).float()
        criterion = torch.nn.MSELoss()
        loss   = criterion(output, target.unsqueeze(1))
        return loss

    class csv_Dataset(Dataset):
    # Here we define a custom Dataset class inherited from the general torch Dataset class
    # This class takes as argument a .csv file path and creates a torch Dataset
        def __init__(self, dataset_path, x_dim):
            self.input_file = pd.read_csv(dataset_path,sep=';',index_col=False)
            x_train = self.input_file.iloc[:,:x_dim].values
            y_train = self.input_file.iloc[:,-1].values
            self.X_train = torch.from_numpy(x_train).float()
            self.Y_train = torch.from_numpy(y_train).float()

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):

            return (self.X_train[idx], self.Y_train[idx])

    def training_data(self,  batch_size = 48):
    # The training_data creates the Dataloader to be used for training in the general class TorchTrainingPlan of fedbiomed
        dataset = self.csv_Dataset(self.dataset_path, self.in_features)
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        data_loader = DataLoader(dataset, **train_kwargs)
        return data_loader


# model parameters
model_args = {
    'in_features': 15,
    'out_features': 1
}

# training parameters
training_args = {
    'batch_size': 20,
    'lr': 1e-3,
    'epochs': 10,
    'dry_run': False,
    #'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}


# Define an experiment
# - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
# - run a round of local training on nodes with model defined in `model_path` + federation with `aggregator`
# - run for `rounds` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

# Calling the training data with specified tags
tags =  ['#test_data']
rounds = 5

exp = Experiment(tags=tags,
                 #nodes=None,
                 model_class=MyTrainingPlan,
                # model_class=AlterTrainingPlan,
                # model_path='/path/to/model_file.py',
                 model_args=model_args,
                 training_args=training_args,
                 rounds=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)


# Let's start the experiment.
# By default, this function doesn't stop until all the `rounds` are done for all the nodes

exp.run()


# Local training results for each round and each node are available in `exp.training_replies` (index 0 to (`rounds` - 1) ).
# For example you can view the training results for the last round below.
#
# Different timings (in seconds) are reported for each dataset of a node participating in a round :
# - `rtime_training` real time (clock time) spent in the training function on the node
# - 'ptime_training` process time (user and system CPU) spent in the training function on the node
# - `rtime_total` real time (clock time) spent in the researcher between sending the request and handling the response, at the `Job()` layer

print("\nList the training rounds : ", exp.training_replies.keys())

print("\nList the nodes for the last training round and their timings : ")
round_data = exp.training_replies[rounds - 1].data()
for c in range(len(round_data)):
    print("\t- {id} :\
    \n\t\trtime_training={rtraining:.2f} seconds\
    \n\t\tptime_training={ptraining:.2f} seconds\
    \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['node_id'],
        rtraining = round_data[c]['timing']['rtime_training'],
        ptraining = round_data[c]['timing']['ptime_training'],
        rtotal = round_data[c]['timing']['rtime_total']))
print('\n')

exp.training_replies[rounds - 1].dataframe()


# Federated parameters for each round are available in `exp.aggregated_params` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", exp.aggregated_params.keys())

print("\nAccess the federated params for the last training round : ")
print("\t- params_path: ", exp.aggregated_params[rounds - 1]['params_path'])
print("\t- parameter data: ", exp.aggregated_params[rounds - 1]['params'].keys())
