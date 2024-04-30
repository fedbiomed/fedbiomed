"""This test file tests launching many nodes and executing an experiment
with 200 rounds of training using also secure aggregation
"""
import time

import pytest

from helpers import (
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_experiment_data,
    clear_researcher_data,
    get_data_folder,
    create_researcher,
    create_multiple_nodes,
    configure_secagg,
    secagg_certificate_registration,
    generate_sklearn_classification_dataset

)

from experiments.training_plans.sklearn import PerceptronTraining
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    configure_secagg()

    print(port)
    with create_multiple_nodes(port, 10) as nodes:

        p1, _, _ = generate_sklearn_classification_dataset()

        dataset = {
            "name": "Classification dataset",
            "description": "ddesc",
            "tags": "#csv-dataset-classification",
            "data_type": "csv",
            "path": p1}

        secagg_certificate_registration()

        for node in nodes:
            print(node)
            add_dataset_to_node(node, dataset)

        researcher = create_researcher(port=port)

        node_processes, thread = start_nodes(list(nodes))

        # Give some time to start nodes in parallel
        time.sleep(30)


        yield tuple(nodes)

        kill_subprocesses(node_processes)
        thread.join()
        print("Clearing component data")
        clear_researcher_data(researcher)

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

per_model_args = {
    'max_iter':1000,
    'tol': 1e-3 ,
    'n_features' : 20,
    'n_classes' : 2
}

per_training_args = {
    'epochs': 5,
    'loader_args': { 'batch_size': 1 }
}



def test_01_sklearn_many_nodes_testing():
    """Tests SGD classifier with Declear optimizers"""

    exp = Experiment(
        tags=['#csv-dataset-classification'],
        model_args=per_model_args,
        training_plan_class=PerceptronTraining,
        training_args=per_training_args,
        round_limit=200,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        save_breakpoints=True,
        secagg=True
        )

    exp.run()

    clear_experiment_data(exp)


