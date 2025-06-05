# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainingPlanStatusRequest
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.requests import DiscardOnTimeout

from ._job import Job


class TrainingPlanApproveJob(Job):
    """Task for requesting nodes approval for running a given
    [TrainingPlan][fedbiomed.common.training_plans.BaseTrainingPlan] on these nodes.
    """

    def __init__(self,
                 training_plan: BaseTrainingPlan,
                 description: str,
                 **kwargs
                 ):
        """Constructor of the class.

        Args:
            training_plan: an instance of a TrainingPlan object
            description: human-readable description of the TrainingPlan for the reviewer on the node
            *args: Positonal argument of parent class
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
            **kwargs: Named arguments of parent class. Please see
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
        """
        super().__init__(**kwargs)
        self._policies = [DiscardOnTimeout(5)]  # specific policy for TrainingApproval
        self._training_plan = training_plan
        self._description = description

    def execute(self) -> Dict:
        """Requests the approval of the provided TrainingPlan.

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the training plan has been correctly downloaded on the node side.
            Warning: status does not mean that the training plan is approved, only that it has been added
            to the "approval queue" on the node side.
        """
        return self._reqs.training_plan_approve(self._training_plan,
                                                self._description,
                                                self._nodes,
                                                self._policies)


class TrainingPlanCheckJob(Job):
    """Task for checking if nodes accept running a given
    [TrainingPlan][fedbiomed.common.training_plans.BaseTrainingPlan].
    """

    def __init__(
        self,
        experiment_id: str,
        training_plan: BaseTrainingPlan,
        **kwargs
    ):
        """Constructor of the class.

        Args:
            experiment_id: unique ID of this experiment
            training_plan: an instance of a TrainingPlan object
            **kwargs: Named arguments of parent class. Please see
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]

        """
        super().__init__(**kwargs)
        self._policies = [DiscardOnTimeout(5)]  # specific policy for TrainingApproval
        self._experiment_id = experiment_id
        self._training_plan = training_plan

    def execute(self) -> Dict:
        """Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        Returns:
            A dict of `Message` objects indexed by node ID, one for each job's nodes
        """

        message = TrainingPlanStatusRequest(**{
            'researcher_id': self._researcher_id,
            'experiment_id': self._experiment_id,
            'training_plan': self._training_plan.source(),
        })

        # Send message to each node that has been found after dataset search request
        # TODO: add timer to compute request time
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
