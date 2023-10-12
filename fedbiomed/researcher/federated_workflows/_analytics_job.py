# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from fedbiomed.common.serializer import Serializer
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.federated_workflows._job import Job
from fedbiomed.researcher.responses import Responses


class AnalyticsJob(Job):
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a temporary file
    (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

    """

    def __init__(self,
                 reqs: Requests = None,
                 nodes: Optional[dict] = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            nodes: A dict of node_id containing the nodes used for training
            training_plan_class: instance or class of the TrainingPlan.
            training_plan_path: Path to file containing model class code
            training_args: Contains training parameters; lr, epochs, batch_size.
            model_args: Contains output and input feature dimension
            data: Federated datasets
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

    def submit_analytics_query(self,
                               query_type: str,
                               data: 'FederatedDataSet',
                               query_kwargs: Optional[dict] = None,
                               secagg_arguments: Optional[dict] = None, ):

        # serialize query arguments
        serialized_query_kwargs = Serializer.dumps(query_kwargs).hex()
        # prepare query request
        msg = {
            'researcher_id': environ["ID"],
            'job_id': self.id,
            'command': 'analytics_query',
            'query_type': query_type,
            'query_kwargs': serialized_query_kwargs,
            'training_plan_url': self._repository_args['training_plan_url'],
            'training_plan_class': self._repository_args['training_plan_class'],
            'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
        }
        # send query request to nodes
        for cli in self.nodes:
            msg['dataset_id'] = data.data()[cli]['dataset_id']
            self.requests.send_message(msg, cli)
        # collect query results from nodes
        replies = Responses(list())
        while self.waiting_for_nodes(replies):
            query_results = self.requests.get_responses(look_for_commands=['analytics_query', 'error'],
                                                        only_successful=False)
            for result in query_results.data():
                result['results'] = Serializer.loads(bytes.fromhex(result['results']))
                replies.append(result)
        return replies

