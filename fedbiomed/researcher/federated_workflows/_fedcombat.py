# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""FedCombat class for harmonizing data within an Experiment using FedCombat algorithm."""

import uuid

from fedbiomed.common.constants import PreprocType
from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.federated_workflows.jobs import PreprocRequestJob
from fedbiomed.researcher.requests import Requests


# TODO: Implement a Base Preprocessing class for common preprocessing functionalities
class FedCombat:
    """
    A class to run FedCombat to harmonize data within an Experiment.
    FedCombat is a federated implementation of the ComBat algorithm for data harmonization.
    """

    def __init__(
        self,
        fds: FederatedDataSet,
        experiment_id: str,
        researcher_id: str,
        reqs: Requests,
        experimentation_folder: str,
        **kwargs,
    ) -> None:
        """Constructor of the class.

        Args:
            **kwargs: Additional named arguments
        """
        self._preproc_id: str = "Preproc_" + str(
            uuid.uuid4()
        )  # creating a unique preprocessing id
        self._fds = fds
        self._experiment_id = experiment_id
        self._researcher_id = researcher_id
        self._reqs = reqs
        self._experimentation_folder = experimentation_folder
        self._kwargs = kwargs

    @property
    def preproc_id(self) -> str:
        """Get the unique ID of this preprocessing job.

        Returns:
            The unique ID of this preprocessing job
        """
        return self._preproc_id

    def get_node_ids(self) -> list[str]:
        """Get the list of node IDs participating in this preprocessing job.

        Returns:
            A list of node IDs
        """
        if self._fds is None:
            raise FedbiomedExperimentError(
                "No defined FederatedDataSet found for FedCombat."
            )

        node_ids = self._fds.node_ids()
        if len(node_ids) == 0:
            raise FedbiomedExperimentError(
                "Empty list of nodes for analytics: no nodes replied to original "
                "`federated_analytics_request` or sampling strategy returned an empty list."
            )

        return node_ids

    def harmonize(self):
        node_ids = self.get_node_ids()

        # Create Preproc job
        preproc_job = PreprocRequestJob(
            preproc_id=self._preproc_id,
            preproc_type=PreprocType.FEDCOMBAT.value,
            preproc_step=0,
            preproc_args={
                "parameters": 0,  # Placeholder for actual FedCombat parameters
            },
            federated_dataset=self._fds,
            experiment_id=self._experiment_id,
            researcher_id=self._researcher_id,
            requests=self._reqs,
            nodes=node_ids,
        )

        # Placeholder for Step 1
        analytics_replies = preproc_job.execute()

        logger.debug(f"Analytics replies after harmonization: {analytics_replies}")

        for node_id, reply in analytics_replies.items():
            preproc_output = reply.preproc_output

            if preproc_output["preproc_step"] != preproc_job._preproc_step + 1:
                raise FedbiomedExperimentError(
                    f"Preprocessing step {preproc_job._preproc_step} failed on node {node_id}"
                )

        preproc_job._preproc_step = preproc_output["preproc_step"] + 1
        preproc_job._preproc_args = {
            "parameters": preproc_output["new_parameters"],
        }

        logger.debug(f"Current step is {preproc_job._preproc_step}")
        logger.debug(
            f"New Fedcombat parameters after harmonization: {preproc_job._preproc_args['parameters']}"
        )

        # Placeholder for Step 2
        analytics_replies = preproc_job.execute()

        logger.debug(f"Analytics replies after harmonization: {analytics_replies}")

        for node_id, reply in analytics_replies.items():
            preproc_output = reply.preproc_output

            if preproc_output["preproc_step"] != preproc_job._preproc_step + 1:
                raise FedbiomedExperimentError(
                    f"Preprocessing step {preproc_job._preproc_step} failed on node {node_id}"
                )

        preproc_job._preproc_step = preproc_output["preproc_step"] + 1
        preproc_job._preproc_args = {
            "parameters": preproc_output["new_parameters"],
        }

        logger.debug(f"Current step is {preproc_job._preproc_step}")
        logger.debug(
            f"New Fedcombat parameters after harmonization: {preproc_job._preproc_args['parameters']}"
        )

        return analytics_replies
