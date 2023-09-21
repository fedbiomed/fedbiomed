# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.job import Job
from fedbiomed.common.serializer import Serializer


class FedAnalytics:
    def __init__(self,
                 job: Job
                 ):
        self._job = job
        self._researcher_id = environ['RESEARCHER_ID']
        self._responses_history = list()
        self._aggregation_results_history = list()

    def fed_mean(self, **kwargs):
        return self._submit_fed_analytics_query(query_type='mean', query_kwargs=kwargs)

    def _submit_fed_analytics_query(self, query_type: str, query_kwargs: dict):

        self._job.nodes = self._job.data.node_ids()

        msg = {
            'researcher_id': self._researcher_id,
            'job_id': self._job.id,
            'command': 'analytics_query',
            'query_type': query_type,
            'query_kwargs': query_kwargs,
            'training_plan_url': self._job._repository_args['training_plan_url'],
            'training_plan_class': self._job._repository_args['training_plan_class']
        }
        for cli in self._job.nodes:
            msg['dataset_id'] = self._job.data.data()[cli]['dataset_id']
            self._job.requests.send_message(msg, cli)  # send request to node

        # Recollect models trained
        self._responses_history.append(Responses(list()))
        while self._job.waiting_for_nodes(self._responses_history[-1]):
            query_results = self._job.requests.get_responses(look_for_commands=['analytics_query', 'error'],
                                                             only_successful=False)
            for result in query_results.data():
                result['results'] = Serializer.loads(bytes.fromhex(result['results']))
                self._responses_history[-1].append(result)

        results = [x['results'] for x in self._responses_history[-1]]
        data_manager = self._job.training_plan.training_data()
        data_manager.load(tp_type=self._job.training_plan.type())
        aggregation_function = getattr(data_manager.dataset, 'aggregate_' + query_type)
        aggregation_result = aggregation_function(results)

        return aggregation_result, {x['node_id']: x['results'] for x in self._responses_history[-1].data()}
