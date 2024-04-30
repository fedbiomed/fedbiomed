"""
Test module to test a big model (60MB) using dry run training
"""
import pytest

from helpers import (
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_experiment_data,
    clear_researcher_data,
    get_data_folder,
    create_researcher,
    create_multiple_nodes
)

from experiments.training_plans.mnist_pytorch_training_plan import BigModelMyTrainingPlan

from fedbiomed.common.metrics import MetricTypes
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""
    data_folder = get_data_folder('MNIST-e2e-test')
    with create_multiple_nodes(port, 3) as nodes:
        dataset = {
            "name": "MNIST",
            "description": "MNIST DATASET",
            "tags": "#MNIST,#dataset",
            "data_type": "default",
            "path": data_folder
        }

        for node in nodes:
            add_dataset_to_node(node, dataset)

        researcher = create_researcher(port=port)
        node_processes, thread = start_nodes(list(nodes))

        yield

        kill_subprocesses(node_processes)
        thread.join()
        print("Clearing component data")
        clear_researcher_data(researcher)

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################



def test_01_mnist_pytorch_big_model_training_dry_run():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 100
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 100,
        'dry_run': True,

    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=BigModelMyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,)

    exp.run()

    clear_experiment_data(exp)

