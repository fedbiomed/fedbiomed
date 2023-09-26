# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses
from fedbiomed.common.serializer import Serializer


class FedAnalytics:
    def __init__(self,
                 exp: 'Experiment'
                 ):
        self._researcher_id = environ['RESEARCHER_ID']
        self._responses_history = list()
        self._aggregation_results_history = list()
        self._exp = exp

    def fed_mean(self, **kwargs):
        return self._submit_fed_analytics_query(query_type='mean', query_kwargs=kwargs)

    def fed_std(self, **kwargs):
        return self._submit_fed_analytics_query(query_type='std', query_kwargs=kwargs)

    def _submit_fed_analytics_query(self, query_type: str, query_kwargs: dict):

        self._exp.job().nodes = self._exp.job().data.node_ids()
        serialized_query_kwargs = Serializer.dumps(query_kwargs).hex()

        # If secure aggregation is activated ---------------------------------------------------------------------
        secagg_arguments = {}
        if self._exp.secagg.active:
            self._exp.secagg.setup(parties=[environ["ID"]] + self._exp.job().nodes,
                               job_id=self._exp.job().id)
            secagg_arguments = self._exp.secagg.train_arguments()
        # --------------------------------------------------------------------------------------------------------

        msg = {
            'researcher_id': self._researcher_id,
            'job_id': self._exp.job().id,
            'command': 'analytics_query',
            'query_type': query_type,
            'query_kwargs': serialized_query_kwargs,
            'training_plan_url': self._exp.job()._repository_args['training_plan_url'],
            'training_plan_class': self._exp.job()._repository_args['training_plan_class'],
            'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
        }
        for cli in self._exp.job().nodes:
            msg['dataset_id'] = self._exp.job().data.data()[cli]['dataset_id']
            self._exp.job().requests.send_message(msg, cli)  # send request to node

        # Recollect models trained
        self._responses_history.append(Responses(list()))
        while self._exp.job().waiting_for_nodes(self._responses_history[-1]):
            query_results = self._exp.job().requests.get_responses(look_for_commands=['analytics_query', 'error'],
                                                             only_successful=False)
            for result in query_results.data():
                result['results'] = Serializer.loads(bytes.fromhex(result['results']))
                self._responses_history[-1].append(result)

        results = [x['results'] for x in self._responses_history[-1]]
        data_manager = self._exp.job().training_plan.training_data()
        data_manager.load(tp_type=self._exp.job().training_plan.type())

        if self._exp.secagg.active:
            flattened = self._exp.secagg.aggregate(
                round_=0,
                encryption_factors={
                    x['node_id']: x['results']['encryption_factor'] for x in self._responses_history[-1]
                },
                total_sample_size=reduce(
                    lambda x,y: x + y['results']['num_samples'],
                    self._responses_history[-1], 0),
                model_params={
                    x['node_id']: x['results']['flat'] for x in self._responses_history[-1]
                }
            )
            unflatten = getattr(data_manager.dataset, 'unflatten')
            aggregation_result = unflatten({
                'flat': flattened,
                'format': results[0]['format']})
        else:
            aggregation_function = getattr(data_manager.dataset, 'aggregate_' + query_type)
            aggregation_result = aggregation_function(results)

        return aggregation_result, {x['node_id']: x['results'] for x in self._responses_history[-1].data()}
