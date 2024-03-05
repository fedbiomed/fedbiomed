import time
import pytest

from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_component_data)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(request):
    """Setup fixture for the module"""

    print("Creating components ---------------------------------------------")
    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }

    print("Adding first dataset --------------------------------------------")
    add_dataset_to_node(node_1, dataset)
    print("adding second dataset")
    add_dataset_to_node(node_2, dataset)

    time.sleep(1)

    # Starts the nodes
    node_processes, _ = start_nodes([node_1, node_2])

    # Clear files and processes created for the tests
    def clear():
        kill_subprocesses(node_processes)

        print("Cleareaniing component data")
        clear_component_data(node_1)
        clear_component_data(node_2)

    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(5)

    request.addfinalizer(clear)

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################



def test_experiment_run_01():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 2
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'epochs': 1,
        'dry_run': False,
        'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None)
    exp.run()



def test_experiment_run_02():
    """Test!"""

    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 2
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'epochs': 1,
        'dry_run': False,
        'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None)

    exp.run()
