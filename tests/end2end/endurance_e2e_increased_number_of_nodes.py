"""This test file tests launching many nodes and executing an experiment
with 200 rounds of training using also secure aggregation
"""

import time

import pytest
from experiments.training_plans.sklearn import PerceptronTraining
from helpers import (
    add_dataset_to_node,
    clear_experiment_data,
    generate_sklearn_classification_dataset,
)

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(federation):
    """Setup fixture for the module"""

    with federation.nodes(10) as nodes:
        p1, _, _ = generate_sklearn_classification_dataset()

        dataset = {
            "name": "Classification dataset",
            "description": "desc",
            "tags": "#csv-dataset-classification",
            "data_type": "csv",
            "path": p1,
        }

        for node in nodes:
            print(node)
            add_dataset_to_node(node, dataset)

        federation.start(nodes)

        # Give some time to start nodes in parallel
        time.sleep(30)

        yield nodes


#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

per_model_args = {"max_iter": 1000, "tol": 1e-3, "n_features": 20, "n_classes": 2}

per_training_args = {"epochs": 5, "loader_args": {"batch_size": 1}}


def test_01_sklearn_many_nodes_testing():
    """Tests SGD classifier with Declear optimizers"""

    exp = Experiment(
        tags=["#csv-dataset-classification"],
        model_args=per_model_args,
        training_plan_class=PerceptronTraining,
        training_args=per_training_args,
        round_limit=200,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        save_breakpoints=True,
        secagg=True,
        retain_full_history=False,
    )

    exp.run()

    clear_experiment_data(exp)
