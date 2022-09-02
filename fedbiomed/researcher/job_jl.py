"""Manage the training part of the experiment."""
import pickle
import uuid
from typing import Callable, Type, Union

from fedbiomed.common.logger import logger
from fedbiomed.common.training_plans import SKLearnTrainingPlan  # noqa
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses


class JobJL(Job):
    def __init__(
        self,
        reqs: Requests = None,
        nodes: dict = None,
        model: Union[Type[Callable], str] = None,
        model_path: str = None,
        training_args: dict = None,
        model_args: dict = None,
        data: FederatedDataSet = None,
        keep_files_dir: str = None,
    ):
        super().__init__(
            reqs,
            nodes,
            model,
            model_path,
            training_args,
            model_args,
            data,
            keep_files_dir,
        )

    def _load_params(self, m):
        logger.info(
            f"Downloading model params after training on {m['node_id']} - from {m['params_url']}"
        )
        _, params_path = self.repo.download_file(
            m["params_url"], "node_params_" + str(uuid.uuid4()) + ".pkl"
        )
        with open(params_path, "rb") as handle:
            params = pickle.load(handle)
        logger.info("Model params downloaded")
        return params, params_path
