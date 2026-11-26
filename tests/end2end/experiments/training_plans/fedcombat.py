import torch
import torch.nn as nn
import torch.nn.functional as F
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.training_plans import TorchTrainingPlan


class FedComBatTrainingPlan(TorchTrainingPlan):
    """Regression on the columns of `generate_fedcombat_dataset`, as in the FedComBat tutorial."""

    def init_model(self, model_args):
        return self.Net(model_args)

    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset"]

    class Net(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.fc1 = nn.Linear(model_args["in_features"], 5)
            self.fc2 = nn.Linear(5, model_args["out_features"])

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    def training_step(self, data, target):
        output = self.model().forward(data).float()
        return torch.nn.MSELoss()(output, target)

    def training_data(self):
        dataset = TabularDataset(
            input_columns=["SEX", "AGE", "PTEDUCAT", "CDRSB.bl"],
            target_columns="RAVLT.forgetting.bl",
            transform=lambda xs: torch.as_tensor(xs, dtype=torch.float32),
            target_transform=lambda xs: torch.as_tensor(xs, dtype=torch.float32),
        )
        return DataManager(dataset=dataset, shuffle=True)
