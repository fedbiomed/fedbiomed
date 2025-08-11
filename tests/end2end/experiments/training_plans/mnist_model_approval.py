import torch
import torch.nn as nn
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.datamanager import DataManager
from torchvision import datasets, transforms


# Here we define the model to be used.
# You can use any class name (here 'Net')
class TrainingPlanApprovalTP(TorchTrainingPlan):
    # Defines and return model
    def init_model(self, model_args):
        return self.Net(model_args=model_args)

    # Defines and return optimizer
    def init_optimizer(self, optimizer_args):
        return torch.optim.Adam(self.model().parameters(), lr=optimizer_args["lr"])

    # Declares and return dependencies
    def init_dependencies(self):
        deps = ["from torchvision import datasets, transforms"]
        return deps

    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)  # out size 14
            self.conv2 = nn.Conv2d(16, 32, 3, 1)  # out size 12
            self.fc1 = nn.Linear(32 * 12 * 12, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)

            output = F.log_softmax(x, dim=1)
            return output

    def training_data(self):
        # Custom torch Dataloader for MNIST data

        #        import sys
        #        sys.exit(12)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset1 = datasets.MNIST(
            self.dataset_path, train=True, download=False, transform=transform
        )
        train_kwargs = {"shuffle": True}
        return DataManager(dataset=dataset1, **train_kwargs)

    def training_step(self, data, target):
        output = self.model().forward(data)
        loss = torch.nn.functional.nll_loss(output, target)
        return loss
