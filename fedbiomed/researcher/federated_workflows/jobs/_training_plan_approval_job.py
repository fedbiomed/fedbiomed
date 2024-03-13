# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainingPlanStatusRequest
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.federated_workflows.jobs._job import Job
from fedbiomed.researcher.requests import DiscardOnTimeout


class TrainingPlanApprovalJob(Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policies = [DiscardOnTimeout(5)]  # specific policy for TrainingApproval

    def check_training_plan_is_approved_by_nodes(self,
                                                 job_id: str,
                                                 training_plan: BaseTrainingPlan,
                                                 ) -> Dict:
        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        Parameters:
            job_id: unique ID of this task
            training_plan: an instance of a TrainingPlan object

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
        with self._reqs.send(message, self._nodes, policies=self._policies) as federated_req:
            replies = federated_req.replies()

            for node_id, reply in replies.items():
                if reply.success is True:
                    if reply.approval_obligation is True:
                        if reply.status == TrainingPlanApprovalStatus.APPROVED.value:
                            logger.info(f'Training plan has been approved by the node: {node_id}')
                        else:
                            logger.warning(f'Training plan has NOT been approved by the node: {node_id}.' +
                                           f'Training plan status : {reply.status}')
                    else:
                        logger.info(f'Training plan approval is not required by the node: {node_id}')
                else:
                    logger.warning(f"Node : {node_id} : {reply.msg}")

        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(self._nodes) - set(replies.keys()))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return replies

    def training_plan_approve(self,
                              training_plan: BaseTrainingPlan,
                              description: str) -> Dict:
        """Requests the approval of the provided TrainingPlan.

        Parameters:
            training_plan: an instance of a TrainingPlan object
            description: human-readable description of the TrainingPlan for the reviewer on the node
        """
        return self._reqs.training_plan_approve(training_plan,
                                                description,
                                                self._policies,
                                                self._nodes)
