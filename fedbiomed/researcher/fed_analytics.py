# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Fed-BioMed Federated Analytics.

This module implements the logic for running audited analytics queries
on the datasets belonging to the federation of Fed-BioMed nodes in an experiment.
"""

from functools import reduce
from typing import Any, Dict, Tuple
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses
from fedbiomed.common.serializer import Serializer


QueryResult = Any  # generic type indicating the result from an analytics query
NodeId = str


class FedAnalytics:
    """
    Federated Analytics API for researcher.

    This class defines the public API for the analytics queries that can be run within the Fed-BioMed federation.
    This class also defines the logic for "translating" an analytics query request by the researcher into a job
    to be executed on the nodes, and for orchestrating said job.

    !!! warning "This class must always be linked to a valid `Experiment`"
        Undefined behaviour may occur if the experiment instance is deleted or rendered invalid in some way. The linked
        `Experiment` instance must be considered **read-only** within the context of `FedAnalytics`. No side-effects
        may modify the linked experiment.

    !!! info "Researcher interface"
        The intention of this class is to provide additional method to `Experiment` without cluttering it. Thus,
        the intended usage of this class is always within the `Experiment` instance it is linked to, such that
        the researcher interface to the query methods looks like this: `exp.analytics.fed_query()`

    Assumptions:

    - an [`Experiment`][fedbiomed.researcher.experiment.Experiment] has been defined and remains valid throughout
      the life-cycle of `FedAnalytics`;
    - the `Experiment` holds a well-defined [`Job`][fedbiomed.researcher.job.Job] as well a well-defined [`FederatedDataSet`][fedbiomed.researcher.datasets.FederatedDataSet];
    - the dataset class corresponding to the data type on the nodes implements the appropriate analytics functions

    Adding a new analytics query
    ===

    First, define a `fed_<query>` function (e.g. `fed_mean`) which calls the protected method
    `_submit_fed_analytics_query`. Then, in order to be compliant with our workflow, the Dataset class must
    implement the following functions:

    | name | description | example |
    | --- | --- | --- |
    | `init` | the dataset must support construction without setting a data path | `TabularDataset.__init__` |
    | `<query>` | takes `query_kwargs` as input and returns a dict of serializable items representing the result of the query | [`TabularDataset.mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.mean] |
    | `aggregate_<query>` | static method that takes the results from each node's query and returns the aggregated values | [`TabularDataset.aggregate_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.aggregate_mean] |

    Algorithm
    ===

    Pseudo-code:

    - when `exp.analytics.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
      is sent to the nodes
    - on the node, a `Round` is instantiated and the `run_analytics_query` method is called
    - each node instantiates the `TrainingPlan` and executes the `training_data` function to obtain a `DataManager`
    - from `DataManager` we obtain the `Dataset` object, on which `Round` calls `Dataset.<query>` to obtain the
      node-specific query results. The format of the results is defined by the `Dataset` class itself
    - The node-specific results are serialized and sent to the aggregator, which deserializes them
    - the aggregator also instantiates the `TrainingPlan` and calls `training_data`. However, the dataset object
      is not linked to any path. We only use static methods from the dataset class on the aggregator
    - the `Dataset.aggregate_<query>` static method is called

    Secure Aggregation
    ===

    Federated analytics queries may support the secure aggregation protocol.
    This will prevent the researcher from accessing the query results from each node, as they will only be able to
    view in clear the aggregated result.

    !!! warning "Limitations"
        The only supported aggregation method is plain (i.e. unweighted) averaging. Thus secure aggregation will only
        yield correct results if the number of samples on each node is the same, and if averaging is the correct
        aggregation operation for a given query.

    To support secure aggregation, the `Dataset` class must also implement:

    | name | description | example |
    | --- | --- | --- |
    | `flatten_<query>` | optional static method that takes the output of <query> and returns a dict with at least two keys: `flat` corresponding to a flattened (1-dimensional) array of results, and `format` with the necessary shape to unflatten the array | [`TabularDataset.flatten_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.flatten_mean] |
    | `unflatten_<query>` | optional static method that takes a flattened output (i.e. a dict with the `flat` and `format` keys) and returns the reconstructed results | [`TabularDataset.unflatten_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.unflatten_mean] |

    The algorithm for executing the federated query is changed as follows in the case of secure aggregation:

    - when `exp.analytics.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
      is sent to the nodes
    - on the node, a `Round` is instantiated and the `run_analytics_query` method is called
    - each node instantiates the `TrainingPlan` and executes the `training_data` function to obtain a `DataManager`
    - from `DataManager` we obtain the `Dataset` object, on which `Round` calls `Dataset.<query>` to obtain the
      node-specific query results. The format of the results is defined by the `Dataset` class itself
    - The node-specific results are flattened and then serialized, before being sent to the aggregator,
      which deserializes them
    - the aggregator also instantiates the `TrainingPlan` and calls `training_data`. However, the dataset object
      is not linked to any path. We only use static methods from the dataset class on the aggregator
    - the secure aggregation method is called. This method only computes the sum of the flattened parameters from each
      node's contribution
    - the summed encrypted weights are decrypted and averaged (unweighted). Finally, the results are unflattened

    """
    def __init__(self,
                 exp: 'Experiment'
                 ):
        """Initialize FedAnalytics by establishing a link to a valid Experiment instance.

        Args:
            exp: an instance of [`Experiment`][fedbiomed.researcher.experiment.Experiment] to be linked to this class
        """
        self._researcher_id: str = environ['RESEARCHER_ID']
        self._responses_history: list = list()
        self._aggregation_results_history: list = list()
        self._exp: 'Experiment' = exp

    def fed_mean(self, **kwargs) -> QueryResult:
        """
        Computes federated mean.

        Args:
            kwargs: any keyword arguments as defined by the corresponding `mean` function implemented in the `Dataset`
                    class

        Returns:
            Results as implemented in the `Dataset` class.
        """
        return self._submit_fed_analytics_query(query_type='mean', query_kwargs=kwargs)

    def fed_std(self, **kwargs) -> QueryResult:
        """
        Computes federated standard deviation.

        Args:
            kwargs: any keyword arguments as defined by the corresponding `std` function implemented in the `Dataset`
                    class

        Returns:
            Results as implemented in the `Dataset` class.
        """
        return self._submit_fed_analytics_query(query_type='std', query_kwargs=kwargs)

    def _submit_fed_analytics_query(self,
                                    query_type: str,
                                    query_kwargs: dict) -> Tuple[QueryResult, Dict[NodeId, QueryResult]]:
        """Helper function executing one round of communication for executing an analytics query on the nodes.

        Args:
            query_type: identifier for the name of the analytics function to be executed on the node. The
                        `Dataset` class must implement a function with the same name
            query_kwargs: keyword arguments to be passed to the query function executed on the nodes

        Returns:
            - the aggregated result
            - a dictionary of {node_id: node-specific results}
        """

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
                round_=1,
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
