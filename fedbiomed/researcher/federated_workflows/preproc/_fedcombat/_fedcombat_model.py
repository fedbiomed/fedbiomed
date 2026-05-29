# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""FedCombat class for harmonizing data within an Experiment using Fed-ComBat algorithm."""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.common.preproc import (
    FedCombatBiasModel,
    FedCombatBiologicalModel,
    FedCombatModelWrapper,
)
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet

# some typing of Experiment class argument without importing the whole Experiment class
# to avoid circular imports
ExperimentFactory = Callable[..., Any]


# Use a PyTorch linear model without bias as the biological model for Fed-ComBat,
# wrapped in a custom wrapper to handle the bias term separately.
class _FedCombatTrainingPlan(TorchTrainingPlan):
    """Training plan for Fed-ComBat harmonization model."""

    def init_model(self, model_args: dict) -> nn.Module:
        return FedCombatModelWrapper(
            biological_model=FedCombatBiologicalModel(
                n_covariates=len(model_args.get("covariates")),
                n_phenotypes=len(model_args.get("phenotypes")),
            ),
            bias_model=FedCombatBiasModel(
                n_phenotypes=len(model_args.get("phenotypes")),
            ),
        )

    def init_optimizer(self, optimizer_args):
        return Adam(self.model().parameters(), lr=optimizer_args["lr"])

    def init_dependencies(self):
        return [
            "from torch.optim import Adam",
            "from fedbiomed.common.dataset import TabularDataset",
            "from fedbiomed.common.preproc import FedCombatBiasModel, FedCombatBiologicalModel, FedCombatModelWrapper",
        ]

    def training_data(self):
        dataset = TabularDataset(
            input_columns=self.model_args()["covariates"],
            target_columns=self.model_args()["phenotypes"],
            transform=lambda xs: torch.as_tensor(xs, dtype=torch.float32),
            target_transform=lambda xs: torch.as_tensor(xs, dtype=torch.float32),
        )
        train_kwargs = {"shuffle": True}
        return DataManager(dataset=dataset, **train_kwargs)

    def training_step(self, data, target):
        output = self.model().forward(data).float()
        loss = nn.MSELoss()(output, target)
        return loss


class _FedCombatTrainModel:
    """Class to handle Fed-ComBat harmonization: train harmonization model."""

    def __init__(
        self,
        experiment_class: ExperimentFactory,
        fds: FederatedDataSet,
        nodes: list[str],
        experimentation_folder: str,
        covariates: Optional[List[Union[str, int]]],
        phenotypes: Optional[List[Union[str, int]]],
        training_args: Optional[Union[TrainingArgs, dict]],
        model_args: Optional[Dict],
        rounds: Optional[int],
    ) -> None:
        """Constructor of the class.

        Args:
            experiment_class: Experiment class factory to be used for the harmonization model training.
            fds: FederatedDataSet instance containing the federated dataset to be harmonized.
            nodes: List of node IDs participating in the harmonization.
            experimentation_folder: Name of the *main* experimentation folder
                to save training data files. A distinct folder name is derived for this
                harmonization and is used as a subdirectory of `config.vars[EXPERIMENTS_DIR])`.
            covariates: List of covariate column names or indices used for harmonization.
            phenotypes: List of phenotype column names or indices used for harmonization.
            training_args: User-provided training arguments for the harmonization model.
            model_args: User-provided model arguments for the harmonization model.
            rounds: Number of federated training rounds for the harmonization model.

        Raises:
            FedbiomedExperimentError: if covariates or phenotypes are not provided and correct.
        """

        self._experiment_class = experiment_class
        self._fds = fds
        self._nodes = nodes
        # Caveat : do not use a `experimentation_folder` name finishing
        #     with numbers ([0-9]+) as this would confuse the last experimentation
        #     detection heuristic by `load_breakpoint`.
        self._experimentation_folder = experimentation_folder + "_fedcombat"

        # Covariates and phenotypes used for harmonization
        self._covariates = covariates or []
        self._phenotypes = phenotypes or []

        # Checks on covariates and phenotypes
        if not isinstance(self._covariates, list) or not isinstance(
            self._phenotypes, list
        ):
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "Covariates and phenotypes must be provided as lists for Fed-ComBat harmonization."
            )
        if len(self._covariates) < 1:
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "At least one covariate must be provided for Fed-ComBat harmonization."
            )
        if len(self._phenotypes) < 1:
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "At least one phenotype must be provided for Fed-ComBat harmonization."
            )
        if not all(
            isinstance(v, int) for v in self._covariates + self._phenotypes
        ) and not all(isinstance(v, str) for v in self._covariates + self._phenotypes):
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "Covariates and phenotypes must all be of the same type, either `str` or `int`."
            )
        _duplicates = [
            v
            for v in set(self._covariates + self._phenotypes)
            if (self._covariates + self._phenotypes).count(v) > 1
        ]
        if any(_duplicates):
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "Covariates and phenotypes must be disjoint lists of unique values. "
                f"Duplicates found: {_duplicates}."
            )

        if isinstance(training_args, TrainingArgs):
            training_args = training_args.dict()
        elif isinstance(training_args, dict):
            training_args = copy.deepcopy(training_args)
        else:
            training_args = {}
        self._training_args = training_args

        _default_training_args = {
            "loader_args": {"batch_size": 16},
            "optimizer_args": {"lr": 0.01},
            "epochs": 1,
            "log_interval": 0,
        }
        self._training_args = {**_default_training_args, **self._training_args}

        self._model_args = copy.deepcopy(model_args) or {}
        self._model_args.update(
            {
                "covariates": self._covariates,
                "phenotypes": self._phenotypes,
            }
        )
        self._rounds = rounds or 3

        self._training_plan_class = _FedCombatTrainingPlan

    def execute(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute harmonization model training.

        Raises:
            FedbiomedExperimentError: if harmonization model initialization or training fails.

        Returns:
            A tuple containing the trained biological model parameters, global bias model parameters, and local bias model parameters by node.
        """

        logger.setPrefix(" \033[1m[Fed-ComBat]\033[0m")
        try:
            experiment = self._experiment_class(
                # aggregator=None,  # Use default aggregator
                # agg_optimizer=None,  # Use default aggregator optimizer
                # node_selection_strategy=None,  # Use default strategy
                round_limit=self._rounds,
                # tensorboard=False,  # Don't use tensorboard graphs
                retain_full_history=False,  # Don't retain intermediate results
                training_plan_class=self._training_plan_class,
                training_args=self._training_args,
                model_args=self._model_args,
                # tags=None,  # Don't search data through tags
                nodes=self._nodes,
                # OK to instantiate a distinct FederatedDataSet for a distinct Experiment
                training_data=self._fds.data(),
                experimentation_folder=self._experimentation_folder,
                # secagg=False # Don't use secure aggregation
                # save_breakpoints=False,  # Don't save breakpoints
                # config_path=None,  # Use default config path
            )
        except FedbiomedError:
            logger.setPrefix("")
            raise
        except Exception as e:
            logger.setPrefix("")
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "Fed-ComBat harmonization model initialization failed."
            ) from e

        try:
            experiment.run()
        except FedbiomedError:
            logger.setPrefix("")
            raise
        except Exception as e:
            logger.setPrefix("")
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: "
                "Fed-ComBat harmonization model training failed."
            ) from e

        # Caveat: only tested with simple liear model but
        # should work with any model as long as the parameter naming
        # convention is respected (i.e. `biological_model.` and `bias_model.` prefixes).
        final_round = experiment.round_current() - 1
        all_model_params = experiment.training_plan().get_model_params()
        biological_model = {
            k.replace("biological_model.", "", 1): v
            for k, v in all_model_params.items()
            if k.startswith("biological_model.")
        }
        global_bias_model = {
            k.replace("bias_model.", "", 1): v
            for k, v in all_model_params.items()
            if k.startswith("bias_model.")
        }

        # Caveat: sending local biases from researcher is not optimal
        # as it may leave room for malicious researcher manipulation.
        # Optimal implementation would save and access local bias models directly on the nodes,
        # but this is not easily doable currently.
        local_bias_models = {
            k: {
                "local_bias_model": {
                    pk.replace("bias_model.", "", 1): pv
                    for pk, pv in v["params"].items()
                    if pk.startswith("bias_model.")
                }
            }
            for k, v in experiment.training_replies()[final_round].items()
        }

        del experiment
        logger.setPrefix("")

        return (biological_model, global_bias_model, local_bias_models)
