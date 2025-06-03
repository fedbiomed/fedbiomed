import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from torchvision import datasets, transforms
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import ScaffoldClientModule


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


# Here we define the model to be used.
# You can use any class name (here 'Net')
class BigModelMyTrainingPlan(TorchTrainingPlan):

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
            self.fc1 = nn.Linear(9216, 2**12)
            self.fc2 = nn.Linear(2**12, 2**9)
            self.fc3 = nn.Linear(2**9, 10)

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
            x = F.relu(x)
            x = self.fc3(x)

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


class MnistModelScaffoldDeclearn(TorchTrainingPlan):
    
    # Defines and return model 
    def init_model(self, model_args):
        return self.Net(model_args = model_args)
    
    # Defines and return a declearn optimizer
    def init_optimizer(self, optimizer_args):
        #return Optimizer(lr=optimizer_args["lr"] ,modules=[AdamModule()],)# regularizers=[FedProxRegularizer()])
        return Optimizer(lr=optimizer_args['lr'], modules=[ScaffoldClientModule()])
    
    # Declares and return dependencies
    def init_dependencies(self):
        deps = ["from torchvision import datasets, transforms",
                "from fedbiomed.common.optimizers.optimizer import Optimizer",
                "from fedbiomed.common.optimizers.declearn import ScaffoldClientModule, AdamModule, FedProxRegularizer"]
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
