import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import ImageFolderDataset, MedNistDataset
from fedbiomed.common.training_plans import TorchTrainingPlan


class MedNistTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        model = models.densenet121(
            weights=None
        )  # here model coefficients are set to random weights

        # add the classifier
        num_classes = model_args["num_classes"]
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        return model

    def init_dependencies(self):
        return [
            "from torchvision import transforms, models",
            "import torch.optim as optim",
            "from torchvision.models import densenet121",
            "from fedbiomed.common.dataset import MedNistDataset",
        ]

    def init_optimizer(self, optimizer_args):
        return optim.Adam(self.model().parameters(), lr=optimizer_args["lr"])

    def training_data(self):
        # Transform images and  do data augmentation
        preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset1 = MedNistDataset(transform=preprocess)
        train_kwargs = {"shuffle": True}
        return DataManager(dataset=dataset1, **train_kwargs)

    def training_step(self, data, target):
        output = self.model().forward(data)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(output, target)
        return loss


class ImageFolderTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        model = models.densenet121(
            weights=None
        )  # here model coefficients are set to random weights

        # add the classifier
        num_classes = model_args["num_classes"]
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        return model

    def init_dependencies(self):
        return [
            "from torchvision import transforms, models",
            "import torch.optim as optim",
            "from torchvision.models import densenet121",
            "from fedbiomed.common.dataset import ImageFolderDataset",
        ]

    def init_optimizer(self, optimizer_args):
        return optim.Adam(self.model().parameters(), lr=optimizer_args["lr"])

    def training_data(self):
        # Transform images and  do data augmentation
        preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset2 = ImageFolderDataset(transform=preprocess)
        train_kwargs = {"shuffle": True}
        return DataManager(dataset=dataset2, **train_kwargs)

    def training_step(self, data, target):
        output = self.model().forward(data)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(output, target)
        return loss
