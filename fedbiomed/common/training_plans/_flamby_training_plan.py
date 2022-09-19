from fedbiomed.common.data import DataManager, FlambyDataset
from fedbiomed.common.training_plans import TorchTrainingPlan


class FlambyTrainingPlan(TorchTrainingPlan):
    def __init__(self, model_args: dict = {}):
        """ Construct training plan

        Args:
            model_args: model arguments. Items used in this class build time
        """
        super().__init__(model_args)
        self.add_dependency(["from fedbiomed.common.training_plans import FlambyTrainingPlan"])

    def training_data(self, batch_size=2):
        """

        If researcher wants to set a transform, they can override the function as follows:
        def training_data(self, batch_size=2):
            data_manager = super().training_data(batch_size)
            transform = Compose([Resize((48,60,48)), NormalizeIntensity()])
            data_manager.dataset.set_transform(myComposedTransform)
            return data_manager
        """
        dataset = FlambyDataset()
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **train_kwargs)
