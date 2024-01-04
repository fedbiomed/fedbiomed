# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainingPlanStatusRequest
from fedbiomed.researcher.federated_workflows.jobs._job import Job
from fedbiomed.researcher.requests import Requests, DiscardOnTimeout


class TrainingPlanApprovalJob(Job):
    def __init__(self,
                 reqs: Requests = None,
                 nodes: Optional[dict] = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            nodes: A dict of node_id containing the nodes used for training
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        Raises:
            NameError: If model is not defined or if the class can not to be inspected
        """

        super().__init__(
            reqs,
            nodes,
            keep_files_dir
        )

    def check_training_plan_is_approved_by_nodes(self,
                                                 job_id: str,
                                                 training_plan,
                                                 ) -> Dict:
        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        Returns:
            A dict of `Message` objects indexed by node ID, one for each job's nodes
        """

        message = TrainingPlanStatusRequest(**{
            'researcher_id': self._researcher_id,
            'job_id': job_id,
            'training_plan': training_plan.source(),
            'command': 'training-plan-status'
        })

        # Send message to each node that has been found after dataset search request
        with self._reqs.send(message, self.nodes, policies=[DiscardOnTimeout(5)]) as federated_req:
            replies = federated_req.replies()

            for node_id, reply in replies.items():
                if reply.success is True:
                    if reply.approval_obligation is True:
                        if reply.status == TrainingPlanApprovalStatus.APPROVED.value:
                            logger.info(f'Training plan has been approved by the node: {node_id}')
                        else:
                            logger.warning(f'Training plan has NOT been approved by the node: {node_id}.' +
                                           f'Training plan status : {node_id}')
                    else:
                        logger.info(f'Training plan approval is not required by the node: {node_id}')
                else:
                    logger.warning(f"Node : {node_id} : {reply.msg}")

        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(self.nodes) - set(replies.keys()))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return replies

