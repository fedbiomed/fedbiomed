#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher to train a variational autoencoder

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
# ## Start the network and setting node up
# 
# Before running this notebook, you should start the network
# Therefore, it is necessary to previously configure a node:
# 
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

import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from torchvision import datasets, transforms


class VariationalAutoencoderPlan(TorchTrainingPlan):
    """ Declaration of two encoding layers and 2 decoding layers
    """
    def init_model(self):
        return self.Net()
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)


        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)


        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))


        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
    
        """ Forward step in variational autoencoders is done in three steps, encoding
        reparametrizing and decoding.
        """
        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
    
    """ We will work on MNIST data. This is the pytorch wrapper of this data.
    """
    def training_data(self,  batch_size = 48):
        # The training_data creates the Dataloader to be used for training in the general class Torchnn of fedbiomed
        mnist_transform = transforms.Compose([
                transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(self.dataset_path, transform=mnist_transform, train=True, download=True)
        return DataManager(train_dataset,batch_size=batch_size,shuffle=True)
    
    """ Computed loss for variational autoencoders.
    """
    def final_loss(self,bce_loss, mu, logvar):
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    """ At each federated learning round, this code will be executed
    in every node making part of the federation.
    """
    def training_step(self, data, target):
       
        criterion = nn.BCELoss(reduction='sum')
        reconstruction, mu, logvar = self.model().forward(data)
        
        bce_loss = criterion(reconstruction, data.view(48,-1))
        loss = self.final_loss(bce_loss, mu, logvar)
        return loss


model_args = {}

training_args = {
    'batch_size': 48,
    'optimizer_args': {
        'lr': 1e-3
    },
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}



#    Define an experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `round_limit` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 3

exp = Experiment(tags=tags,
                 model_args=model_args,
                 training_plan_class=VariationalAutoencoderPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)

# Let's start the experiment.
# By default, this function doesn't stop until all the `round_limit` rounds are done for all the nodes

exp.run()
