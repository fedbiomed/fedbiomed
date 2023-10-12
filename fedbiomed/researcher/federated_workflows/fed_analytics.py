# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Fed-BioMed Federated Analytics.

This module implements the logic for running audited analytics queries
on the datasets belonging to the federation of Fed-BioMed nodes in an experiment.
"""

from functools import reduce
from typing import Any, Dict, List, Tuple, TypeVar, Union
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.secagg import SecureAggregation
from fedbiomed.researcher.federated_workflows._federated_workflow import FederatedWorkflow, Type_TrainingPlan
from fedbiomed.researcher.federated_workflows._analytics_job import AnalyticsJob

QueryResult = Any  # generic type indicating the result from an analytics query
NodeId = str
TDataset = TypeVar('FedbiomedDataset')  # to be defined
TResponses = TypeVar('Responses')  # fedbiomed.researcher.responses.Responses


class FederatedAnalytics(FederatedWorkflow):
    """
    Decorator providing federated analytics API for researcher.

    This decorator defines the public API for the analytics queries that can be run within the Fed-BioMed federation.
    This decorator also defines the logic for "translating" an analytics query request by the researcher into a job
    to be executed on the nodes, and for orchestrating said job.

    Assumptions:

    - the `Experiment` holds a well-defined [`Job`][fedbiomed.researcher.job.Job] as well a well-defined [`FederatedDataSet`][fedbiomed.researcher.datasets.FederatedDataSet];
    - the dataset class corresponding to the data type on the nodes implements the appropriate analytics functions

    The decorator adds the following attributes to the `Experiment` class.

    Attributes:
       _analytics_responses_history (list): a record of all successful query responses from nodes
       _aggregation_results_history (list): a record of all aggregation results

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

    - when `exp.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
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

    - when `exp.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
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

    def __init__(
        self,
        tags: Union[List[str], str, None] = None,
        nodes: Union[List[str], None] = None,
        training_data: Union[FederatedDataSet, dict, None] = None,
        training_plan_class: Union[Type_TrainingPlan, str, None] = None,
        training_plan_path: Union[str, None] = None,
        training_args: Union[TrainingArgs, dict, None] = None,
        experimentation_folder: Union[str, None] = None,
        secagg: Union[bool, SecureAggregation] = False,
    ):
        super().__init__(
            tags,
            nodes,
            training_data,
            training_plan_class,
            training_plan_path,
            training_args,
            experimentation_folder,
            secagg
        )
        self._analytics_responses_history: List[TResponses] = list()
        self._aggregation_results_history: List[Tuple[QueryResult, Dict[NodeId, QueryResult]]] = list()

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
        # setup secagg
        secagg_arguments = self.secagg_setup()
        # sample all nodes
        self.set_nodes(self._fds.node_ids())
        # create AnalyticsJob
        job = AnalyticsJob(
            reqs=self._reqs,
            nodes=self._nodes,
            keep_files_dir=self.experimentation_path()
        )
        self._training_plan = job.create_skeleton_workflow_instance_from_path(self._training_plan_path,
                                                                              self._training_plan_class)
        job.upload_workflow_code(self.training_plan())
        replies = job.submit_analytics_query(
            query_type=query_type,
            query_kwargs=query_kwargs,
            data=self._fds,
            secagg_arguments=secagg_arguments
        )
        self._analytics_responses_history.append(replies)
        # parse results
        results = [x['results'] for x in self._analytics_responses_history[-1]]
        # prepare data manager (only static methods from the dataset can be used)
        dataset_class = self.training_plan().dataset_class
        # aggregate results
        if self.secagg.active:
            aggregation_result = self._secure_aggregate(query_type, results, dataset_class)
        else:
            aggregation_function = getattr(dataset_class, 'aggregate_' + query_type)
            aggregation_result = aggregation_function(results)
        # combine aggregated and node-specific results
        combined_result = (
            aggregation_result,
            {x['node_id']: x['results'] for x in self._analytics_responses_history[-1].data()}
        )
        # store combined results in history
        self._aggregation_results_history.append(combined_result)
        return combined_result

    def _secure_aggregate(self,
                          query_type: str,
                          results: List[QueryResult],
                          dataset_class: TDataset) -> QueryResult:
        """Computes secure aggregation of analytics query results from each node.

        !!! warning "Limitations"
            The only supported aggregation method is plain (i.e. unweighted) averaging. Thus secure aggregation will
            only yield correct results if the number of samples on each node is the same, and if averaging is the
            correct aggregation operation for a given query.

        Args:
            results: the list of query results from each node
            data_manager: the data manager class obtained from the training plan
        """
        # compute average of flattened query results
        flattened = self.secagg.aggregate(
            round_=1,
            encryption_factors={
                x['node_id']: x['results']['encryption_factor'] for x in self._analytics_responses_history[-1]
            },
            total_sample_size=reduce(
                lambda x,y: x + y['results']['num_samples'],
                self._analytics_responses_history[-1], 0),
            model_params={
                x['node_id']: x['results']['flat'] for x in self._analytics_responses_history[-1]
            }
        )
        # unflatten aggregated results
        unflatten = getattr(dataset_class, 'unflatten_' + query_type)
        return unflatten({
            'flat': flattened,
            'format': results[0]['format']})
