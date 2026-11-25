"""Small deterministic training plan shared by both protocol test directions."""

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.training_plans import FedPerceptron


class CompatibilityTrainingPlan(FedPerceptron):
    def init_dependencies(self):
        return ["from fedbiomed.common.dataset import TabularDataset"]

    def training_data(self):
        return DataManager(
            dataset=TabularDataset(input_columns=[0, 1], target_columns=2),
            shuffle=False,
        )
