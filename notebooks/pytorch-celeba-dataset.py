#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher

# This example demonstrates using a convolutional model in PyTorch
# for recognition of smiling faces, with a CelebA dataset split over 2 nodes.

# ## Setting the node up
#
# Install the CelebA dataset with the help of the `README.md` file inside `notebooks/data/Celeba`
# The script create 3 nodes with each their data. The dataset of the node 3 is used in this notebook as a testing set.  
# Therefore its not necessary to create a node and run the node 3  
# 
# Before using script make sure the correct environment is setup, for the node environement, run : `source ./scripts/fedbiomed_environment node`  
# For the sake of testing the resulting model, this file uses the data from node 1 and 2 for training and the data from node 3 to test.
# You can create multiple nodes by adding a config parameter to the command controlling nodes, for example :  
# creating 2 nodes for training :  
#  - `./scripts/fedbiomed_run node config node1.ini start`
#  - `./scripts/fedbiomed_run node config node2.ini start`  
#  
# adding data for each node :  
#  - `./scripts/fedbiomed_run node config node1.ini add`
#  - `./scripts/fedbiomed_run node config node2.ini add`
# 
# It is necessary to previously configure at least 1 node:
# 1. `./scripts/fedbiomed_run node config (ini file) add`
#   * Select option 3 (images) to add an image dataset to the node
#   * Add a name (eg: 'celeba') and the tag for the dataset (tag should contain '#celeba' as it is the tag used for this training) and finaly add the description
#   * Pick a data folder from the 3 generated inside `data/Celeba/celeba_preprocessed` (eg: `data_node_1`)
#   * Data must have been added (if you get a warning saying that data must be unique is because it's been already added)
#   
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node config (ini file) list`
# 3. Run the node using `./scripts/fedbiomed_run node config (ini file) start`. Wait until you get `Starting task manager`. it means you are online.
# 
# For the sake of testing the resulting model, only nodes 1 and 2 were started during training, datas from node 3 is used to test the model.


# ## Create an experiment to train a model on the data found



# Declare a TorchTrainingPlan MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from fedbiomed.common.data import DataManager
import pandas as pd
import numpy as np
from PIL import Image
import os


class CelebaTrainingPlan(TorchTrainingPlan):
         
    # Defines model 
    def init_model(self):
        model = self.Net()
        return model 
    
    # Here we define the custom dependencies that will be needed by our custom Dataloader
    def init_dependencies(self):
        deps = ["from torch.utils.data import Dataset",
                "from torchvision import transforms",
                "import pandas as pd",
                "from PIL import Image",
                "import os",
                "import numpy as np"]
        return deps

    # Torch modules class
    class Net(nn.Module):
        
        def __init__(self):
            super().__init__()
            #convolution layers
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 32, 3, 1)
            self.conv3 = nn.Conv2d(32, 32, 3, 1)
            self.conv4 = nn.Conv2d(32, 32, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            # classifier
            self.fc1 = nn.Linear(3168, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = self.conv3(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = self.conv4(x)
            x = F.max_pool2d(x, 2)
            x = F.relu(x)

            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)

            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


    class CelebaDataset(Dataset):
        """Custom Dataset for loading CelebA face images"""
        
        # we dont load the full data of the images, we retrieve the image with the get item. 
        # in our case, each image is 218*178 * 3colors. there is 67533 images. this take at leas 7G of ram
        # loading images when needed takes more time during training but it wont impact the ram usage as much as loading everything
        def __init__(self, txt_path, img_dir, transform=None):
            df = pd.read_csv(txt_path, sep="\t", index_col=0)
            self.img_dir = img_dir
            self.txt_path = txt_path
            self.img_names = df.index.values
            self.y = df['Smiling'].values
            self.transform = transform
            print("celeba dataset finished")

        def __getitem__(self, index):
            img = np.asarray(Image.open(os.path.join(self.img_dir,
                                        self.img_names[index])))
            img = transforms.ToTensor()(img)
            label = self.y[index]
            return img, label

        def __len__(self):
            return self.y.shape[0]
    
    # The training_data creates the Dataloader to be used for training in the 
    # general class Torchnn of fedbiomed
    def training_data(self,  batch_size = 48):
        dataset = self.CelebaDataset(self.dataset_path + "/target.csv", self.dataset_path + "/data/")
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **train_kwargs)
    
    # This function must return the loss to backward it 
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
    'loader_args': { 'batch_size': 32, },
    'optimizer_args': {
        'lr': 1e-3
    }, 
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}

# # Train the federated model

#    Define an experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of node ID with `nodes`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `round_limit` rounds, applying the `node_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#celeba']
rounds = 3

exp = Experiment(tags=tags,
                 training_plan_class=CelebaTrainingPlan,
                 training_args=training_args,
                 round_limit=rounds,
                 aggregator=FedAverage(),
                 node_selection_strategy=None)


# Let's start the experiment.
# By default, this function doesn't stop until all the `round_limit` rounds are done for all the nodes

exp.run()

# Retrieve the federated model parameters

fed_model = exp.training_plan().model()
fed_model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
print(fed_model)
# # Test Model
#
# We define a little testing routine to extract the accuracy metrics on the testing dataset
# 
# ## Important
#
# This is done to test the model because it can be accessed in a developpement environment
# In production, the data wont be accessible on the nodes, need a test dataset on the server or accessible from the server.

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os

def testing_Accuracy(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    device = "cpu"

    correct = 0
    
    loader_size = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            #only uses 10% of the dataset, results are similar but faster
            if idx >= loader_size / 10:
                pass
                break

    
        pred = output.argmax(dim=1, keepdim=True)

    test_loss /= len(data_loader.dataset)
    accuracy = 100* correct/(data_loader.batch_size * idx)

    return(test_loss, accuracy)

# The test dataset is the data from the third node

print("testing accuracy")
script_path = os.path.dirname(os.path.realpath(__file__))
test_dataset_path = script_path + "/data/Celeba/celeba_preprocessed/data_node_3"

class CelebaDataset(Dataset):
        """Custom Dataset for loading CelebA face images"""

        def __init__(self, txt_path, img_dir, transform=None):
            df = pd.read_csv(txt_path, sep="\t", index_col=0)
            self.img_dir = img_dir
            self.txt_path = txt_path
            self.img_names = df.index.values
            self.y = df['Smiling'].values
            self.transform = transform
            #print("celeba dataset finished")

        def __getitem__(self, index):
            img = np.asarray(Image.open(os.path.join(self.img_dir,
                                        self.img_names[index])))
            img = transforms.ToTensor()(img)
            label = self.y[index]
            return img, label

        def __len__(self):
            return self.y.shape[0]


dataset = CelebaDataset(test_dataset_path + "/target.csv", test_dataset_path + "/data/")
train_kwargs = {'batch_size': 128, 'shuffle': True}
data_loader = DataLoader(dataset, **train_kwargs)


# Loading the testing dataset and computing accuracy metrics for local and federated models


acc_federated = testing_Accuracy(fed_model, data_loader)



print(f"model accuracy on testing set : {acc_federated[1]}")

