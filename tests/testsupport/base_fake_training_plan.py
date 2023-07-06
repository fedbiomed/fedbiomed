from typing import Any, Dict, Optional
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan


class BaseFakeTrainingPlan(TorchTrainingPlan):

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def init_model(self):
        pass

    def init_optimizer(self):
        pass

    def training_data(self):
        pass

    def training_step(self):
        pass
