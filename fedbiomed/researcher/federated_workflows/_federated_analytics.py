import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from fedbiomed.researcher.federated_workflows._experiment import Experiment

from fedbiomed.common.exceptions import FedbiomedExperimentError
from fedbiomed.common.logger import logger
from fedbiomed.researcher.federated_workflows._federated_workflow import exp_exceptions
from fedbiomed.researcher.federated_workflows.jobs._fa_researcher_job import (
    FAResearcherJob,
)


class FederatedAnalytics:
    """
    A Federated Learning Experiment based on a Training Plan.

    This class provides a comprehensive entry point for the management and orchestration
    of a FL experiment, including definition, execution, and interpretation of results.

    !!! note "Managing model parameters"
        The model parameters should be managed through the corresponding methods in the training_plan by accessing
        the experiment's
        [`training_plan()`][fedbiomed.researcher.federated_workflows.TrainingPlanWorkflow.training_plan]
        attribute and using the
        [`set_model_params`][fedbiomed.common.training_plans._base_training_plan.BaseTrainingPlan.set_model_params] and
        [`get_model_params`][fedbiomed.common.training_plans._base_training_plan.BaseTrainingPlan.get_model_params]
        functions, e.g.
        ```python
        exp.training_plan().set_model_params(params_dict)
        ```

    !!! warning "Do not set the training plan attribute directly"
        Setting the `training_plan` attribute directly is not allowed. Instead, use the
        [`set_training_plan_class`][fedbiomed.researcher.federated_workflows.TrainingPlanWorkflow.set_training_plan_class]
        method to set the training plan type, and the underlying model will be correctly
        constructed and initialized.
    """

    @exp_exceptions
    def __init__(
        self,
        experiment: "Experiment",
        **kwargs,
    ) -> None:
        """Constructor of the class.

        Args:
            experiment: An Experiment object to be managed by this workflow
            **kwargs: Additional named arguments
        """
        self._exp = experiment
        self._fa_id: str = "FA_" + str(uuid.uuid4())  # creating a unique experiment id

    @property
    def fa_id(self) -> str:
        """Get the unique ID of this federated analytics.

        Returns:
            The unique ID of this federated analytics
        """
        return self._fa_id

    @property
    def experiment(self) -> "Experiment":
        """Get the Experiment managed by this workflow.

        Returns:
            The Experiment object
        """
        return self._exp

    def mean(self, col_names: Optional[list[str | int]]) -> Union[Any, Dict[str, Any]]:
        """Compute mean analytics across nodes.

        Returns:
            A dictionary containing the mean analytics results from each node.
        """

        # Sample nodes for training
        node_ids = self._exp._node_selection_strategy.sample_nodes(
            from_nodes=self._exp.filtered_federation_nodes(),
            round_i=self._exp._round_current,
        )
        if len(node_ids) == 0:
            raise FedbiomedExperimentError(
                "Empty list of nodes for analytics: no nodes replied to original "
                "`federated_analytics_request` or sampling strategy returned an empty list."
            )

        # Create FA job
        fa_job = FAResearcherJob(
            fa_id=self._fa_id,
            experiment_id=self._exp._experiment_id,
            data=self._exp._fds,
            fa_args={
                "analytics_method": "mean",
                "col_names": col_names if col_names is not None else [],
            },
            node_ids=node_ids,
            researcher_id=self._exp._researcher_id,
            requests=self._exp._reqs,
            nodes=node_ids,
            keep_files_dir=self._exp.experimentation_path(),
        )

        logger.info("Nodes replied for analytics: " + str(fa_job.nodes))

        # Collect training replies and (opt.) optimizer auxiliary variables.
        analytics_replies = fa_job.execute()

        return analytics_replies
