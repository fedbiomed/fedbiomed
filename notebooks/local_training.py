
from fedbiomed.researcher.environ import TMP_DIR

import torch
import torch.nn as nn
from fedbiomed.common.torchnn import TorchTrainingPlan
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fedbiomed.researcher.job import localJob
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


from torchvision import datasets, transforms
import numpy as np

#Declare a torch.nn MyTrainingPlan class to send for training on the node
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
        # Custom torch Dataloader for MNIST data
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        return data_loader

    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss

# This group of arguments correspond respectively:
# * `model_args`: a dictionary with the arguments related to the model (e.g. number of layers, features, etc.). This will be passed to the model class on the client side.
# * `training_args`: a dictionary containing the arguments for the training routine (e.g. batch size, learning rate, epochs, etc.). This will be passed to the routine on the client side.

# **NOTE:** typos and/or lack of positional (required) arguments will raise error.


training_args = {
    'batch_size': 48, 
    'lr': 1e-3, 
    'epochs': 1, 
    'dry_run': False,  
    'batch_maxnum': 200 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}

# Train the federated model

print(' *** Starting federated training *** ')

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                 model_class=MyTrainingPlan,
                 training_args=training_args,
                 rounds=rounds,
                 aggregator=FedAverage(),
                 client_selection_strategy=None)

exp.run()

# Retrieve the federated model parameters and get the federated model

fed_model = exp.model_instance
fed_model.load_state_dict(exp.aggregated_params[rounds - 1]['params'])

print(' *** Federated training complete *** \nFederated model:\n',fed_model)

## Train the local model

print(' *** Starting local training *** ')

# Load the local MNIST training dataset
print('-- Reading local dataset')
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

datasets.MNIST(root = TMP_DIR + '/local_mnist.tmp', download = True, train = True, transform = transform)

# local train on same amount of data as federated with 1 node
training_args['epochs'] *= rounds

# We create an object localJob, which mimics the functionalities of the class Job to run the model on the input local dataset

local_job = localJob( dataset_path = TMP_DIR + '/local_mnist.tmp',
          model_class=MyTrainingPlan,
          #model_path=model_file,
          training_args=training_args)
 
# Running the localJob and getting the local model

local_job.start_training()

local_model = local_job.model_instance

print(' *** Local training complete *** \nLocal model:\n',local_model)

## Comparison

# We define a little testing routine to extract the accuracy metrics on the testing dataset 

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

test_set = datasets.MNIST(root = TMP_DIR + '/local_mnist_testing.tmp', download = True, train = False, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

acc_local = testing_Accuracy(local_model, test_loader)
acc_federated = testing_Accuracy(fed_model, test_loader)

print('\nAccuracy local training: {:.4f}, \nAccuracy federated training:  {:.4f}\nDifference: {:.4f}'.format(
             acc_local[1], acc_federated[1], acc_local[1]-acc_federated[1]))

print('\nError local training: {:.4f}, \nError federated training:  {:.4f}\nDifference: {:.4f}'.format(
             acc_local[0], acc_federated[0], acc_local[0]-acc_federated[0]))

minscore = 0.95
assert  acc_local[1] > minscore and acc_federated[1] > minscore and np.abs(acc_local[1]-acc_federated[1]) < 3.0 

