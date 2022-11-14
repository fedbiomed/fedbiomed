"""Basic example of a custom TorchTrainingPlan.

This example makes a compromise between the revised TrainingPlan API
and the legacy one, hacking around the former to produce lighter and
more human-readable dump files, in the spirit of the latter.
"""

import json
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from fedbiomed.common.data import DataManager
from fedbiomed.common.training_plans import TorchTrainingPlan


class MyTorchTrainingPlan(TorchTrainingPlan):
    """Custom torch training plan, implemented for testing purposes.

    This TorchTrainingPlan:
    * overrides model-creation behaviour, to lighten dump files
    * implements a fake training_data method (returning a mock object)
    """

    class MyTorchModule(nn.Module):
        """Custom torch neural network."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            return F.log_softmax(x, dim=1)

    def __init__(self, model=None, optim=None) -> None:
        """Instantiate the training plan, forcing the model choice."""
        super().__init__(
            model=self.MyTorchModule(),
            optim=optim or {"lrate": 0.001},
            loss=torch.nn.CrossEntropyLoss(),
        )
        self.add_dependency([
            "import json",
            "from unittest import mock",
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "from fedbiomed.common.data import DataManager",
            "from fedbiomed.common.training_plans import TorchTrainingPlan",
        ])

    def save_to_json(self, path) -> None:
        # Override parent method to exclude torch module pickle.
        super().save_to_json(path)
        with open(path, "r", encoding="utf-8") as file:
            dump = json.load(file)
        dump["model"] = None
        with open(path, "w", encoding="utf-8") as file:
            json.dump(dump, file)

    def training_data(self, dataset_path: str) -> DataManager:
        """Return a mock DataManager."""
        manager = mock.create_autospec(DataManager)(
            dataset=mock.MagicMock(),
            target=mock.MagicMock(),
        )
        return manager
