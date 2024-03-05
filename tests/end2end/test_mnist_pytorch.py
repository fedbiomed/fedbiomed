import time
import pytest

from helpers import create_component, add_dataset_to_node, start_nodes
from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


node_processes = None

#@pytest.fixture(scope="module", autouse=True)
#def teardown(request):
#    """Tear down"""
#    def clear():
#        node_processes.terminate()
#    request.addfinalizer(clear)
#


# Set up nodes and start
def setup_custom():
    """Setup fixture for the module"""

    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }

    add_dataset_to_node(node_1, dataset)
    add_dataset_to_node(node_2, dataset)


    # Starts the nodes
    node_processes = start_nodes([node_1, node_2])

    # Good to waiat 3 second to give time to nodes start
    time.sleep(3)

    return node_processes


# Setup testing environment
node_processes = setup_custom()


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


    node_processes.terminate()


test_experiment_run_01()


#def test_experiment_run_02():
#    """Test!"""
#
#    model_args = {}
#    tags = ['#MNIST', '#dataset']
#    rounds = 2
#    training_args = {
#        'loader_args': { 'batch_size': 48, },
#        'optimizer_args': {
#            "lr" : 1e-3
#        },
#        'epochs': 1,
#        'dry_run': False,
#        'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
#    }
#
#    exp = Experiment(
#        tags=tags,
#        model_args=model_args,
#        training_plan_class=MyTrainingPlan,
#        training_args=training_args,
#        round_limit=rounds,
#        aggregator=FedAverage(),
#        node_selection_strategy=None)
#
#    exp.run()
#



