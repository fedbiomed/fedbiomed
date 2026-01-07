# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""FedCombat class for harmonizing data within an Experiment using Fed-ComBat algorithm."""

import uuid
from typing import Any, Optional

from fedbiomed.common.constants import HarmonizationStep, PreprocType
from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows.jobs import PreprocRequestJob
from fedbiomed.researcher.requests import Requests


class FedCombatPreproc:
    """
    A class to run Fed-ComBat to harmonize data of a federated dataset within an experiment.

    Fed-ComBat is a federated implementation of the ComBat algorithm for data harmonization.
    """

    def __init__(
        self,
        fds: FederatedDataSet,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        nodes: list[str],
        experimentation_folder: str,
        preproc_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Constructor of the class.

        Args:
            fds: FederatedDataSet instance containing the federated dataset to be harmonized.
            experiment_id: The experiment ID associated with this Fed-ComBat harmonization.
            researcher_id: The researcher ID initiating this Fed-ComBat harmonization.
            reqs: Requests instance to handle communication with nodes.
            nodes: List of node IDs participating in the harmonization.
            experimentation_folder: Path to the experimentation folder to save harmonization data files.
            preproc_args: Optional dictionary of preprocessing arguments for Fed-ComBat.
        """
        self._preproc_id: str = "preproc_" + str(
            uuid.uuid4()
        )  # creating a unique preprocessing id
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._nodes = (
            nodes  # Caveat: some nodes from FDS may not participate in harmonization
        )
        self._experimentation_folder = experimentation_folder
        self._preproc_args = preproc_args or {}

        self._init_harmonization()

    def _init_harmonization(self) -> None:
        """Initialize harmonization context."""
        # Indicates whether harmonization must be done with current context
        self._do_harmonization: bool = True
        # Context of last harmonization, dict with keys: node ID, values: dataset ID
        self._harmonized_datasets: Optional[dict[str, str]] = None

    def _update_harmonization_done(self) -> None:
        """Update harmonization context after a harmonization is performed."""
        self._do_harmonization = False
        self._harmonized_datasets = {
            node_id: self._fds.data()[node_id]["dataset_id"] for node_id in self._nodes
        }

    def _needs_harmonization(self) -> bool:
        """Determine whether harmonization is needed based on current context.

        Harmonization is needed if:
        - No previous harmonization has been done.
        - The set of nodes has changed since the last harmonization.
        - The federated dataset has changed since the last harmonization.

        Returns:
            bool: True if harmonization is needed, False otherwise.
        """
        if self._do_harmonization:
            return True

        # Check if the set of nodes has changed
        if set(self._harmonized_datasets.keys()) != set(self._nodes):
            return True

        # Check if the dataset IDs have changed for any node
        if any(
            self._harmonized_datasets.get(node_id)
            != self._fds.data()[node_id]["dataset_id"]
            for node_id in self._nodes
        ):
            return True

        return False

    def execute(self) -> bool:
        """Execute Fed-ComBat harmonization across the federated dataset for this experiment.

        Harmonization is specific to the experiment context which includes the set of nodes,
        the federated dataset and the harmonization parameters.

        If harmonization is not needed because it has already been performed and context did not
        change, this method does nothing and returns False.

        After harmonization completes, updates the federated dataset in place and returns True.

        Returns:
            bool: True if harmonization was performed, False otherwise.
        """
        if not self._needs_harmonization():
            return False
        logger.info(
            "Starting Fed-ComBat harmonization for experiment "
            f"{self._experiment_id} with nodes {self._nodes} "
        )

        if len(self._nodes) == 0:
            raise FedbiomedExperimentError(
                "Empty list of nodes for Fed-ComBat: no nodes replied to original "
                "request or sampling strategy returned an empty list."
            )

        for preproc_step in range(1, 7):
            if preproc_step == 1:
                logger.debug(
                    "Starting FedCombat preprocessing step 1: use federated analytics "
                    "to compute global means and variances"
                )
                continue
            elif preproc_step == 3:
                logger.debug(
                    "Starting FedCombat preprocessing step 3: train harmonization model"
                )
                continue

            logger.debug(f"Starting FedCombat preprocessing step {preproc_step}")
            preproc_job = PreprocRequestJob(
                experiment_id=self._experiment_id,
                preproc_type=PreprocType(PreprocType.FEDCOMBAT),
                preproc_step=HarmonizationStep(
                    preproc_step
                ),  # dummy computation for now to remember to derive proper step value
                preproc_id=self._preproc_id,
                federated_dataset=self._fds,
                preproc_args={
                    "dummy": f"Preprocessing step {preproc_step}",
                },
                state_id=None,  # dont have state_id for now
                researcher_id=self._researcher_id,
                requests=self._reqs,
                nodes=self._nodes,
            )
            fedcombat_replies = preproc_job.execute()

            logger.debug(
                f"Fed-ComBat replies after harmonization step {preproc_step}: {fedcombat_replies}"
            )

            # dont check replies for now, to be implemented properly later

        logger.info(
            "Fed-ComBat harmonization completed successfully. Updating federated dataset."
        )
        # TODO: actually update the federated dataset with harmonized dataset IDs
        # self._fds.set_federated_dataset(dict_harmonized_datasets)

        self._update_harmonization_done()
        return True
