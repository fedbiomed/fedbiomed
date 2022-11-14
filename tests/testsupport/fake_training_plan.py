"""Fake fedbiomed TrainingPlan, wrapping a FakeDeclearnModel."""

import time

import declearn

from fedbiomed.common.data import DataLoaderTypes
from fedbiomed.common.training_plans import TrainingPlan


class FakeTrainingPlan(TrainingPlan):
    """Fake fedbiomed TrainingPlan, wrapping a FakeDeclearnModel.

    Note: calling an instance's `replace_training_with_sleep` method
    enables replacing the (complex) base training routine code with
    `time.sleep` for a given delay parameters (in seconds).
    """

    class FakeDeclearnModel(declearn.model.api.Model):
        """Fake declearn Model, exposing minimal effect-less API methods."""

        @property
        def required_data_info(self):
            return set()

        def initialize(self, data_info):
            pass

        def get_config(self):
            return {"model": None}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def get_weights(self):
            return declearn.model.api.NumpyVector({})

        def set_weights(self, weights):
            pass

        def compute_batch_gradients(self, batch):
            return declearn.model.api.NumpyVector({})

        def apply_updates(self):
            pass

        def compute_loss(self, dataset):
            return 0.

    _model_cls=FakeDeclearnModel
    _data_type=DataLoaderTypes.NUMPY

    def __init__(
            self,
            model=None,
            optim={"lrate": 0.01},  # NOTE: minimal config for Optimizer
            **kwargs
        ):
        super().__init__(model, optim, **kwargs)
        self.add_dependency([
            "import time",
            "import declearn",
            "from fedbiomed.common.data import DataLoaderTypes",
            "from fedbiomed.common.training_plans import TrainingPlan",
        ])
        self.delay = None

    def predict(self, data):
        pass  # NOTE: this does *not* respect the API

    def replace_training_with_sleep(self, delay=1):
        """Replace the training routine with `time.sleep(delay)`.

        Set `delay` to None to restore base training routine code use.
        """
        self.delay = delay

    def training_routine(self, *args, **kwargs):
        if self.delay is None:
            super().training_routine(*args, **kwargs)
        else:
            time.sleep(self.delay)
