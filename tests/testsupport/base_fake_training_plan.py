from fedbiomed.common.training_plans import TorchTrainingPlan


class BaseFakeTrainingPlan(TorchTrainingPlan):

    def init_model(self):
        pass

    def init_optimizer(self):
        pass

    def training_data(self):
        pass

    def training_step(self):
        pass
