import os
import time
import pytest

from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_node_data,
    clear_experiment_data,
    clear_researcher_data)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    print(f"USING PORT {port} for researcher erver")
    print("Creating components ---------------------------------------------")
    node_1 = create_component(
        ComponentType.NODE,
        config_name="config_n1_mnist_pytorch.ini",
        config_sections={'researcher': {'port': port}}
    )
    node_2 = create_component(
        ComponentType.NODE,
        config_name="config_n2_mnist_pytorch.ini",
        config_sections={'researcher': {'port': port}}
    )


    researcher = create_component(
        ComponentType.RESEARCHER,
        config_name="config_researcher_mnist_pytorch.ini",
        config_sections={'server': {'port': port}},
    )
    os.environ['RESEARCHER_CONFIG_FILE'] = researcher.name

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
    node_processes, thread = start_nodes([node_1, node_2])

    # Clear files and processes created for the tests
    def clear():
        kill_subprocesses(node_processes)
        thread.join()
        print("Clearing component data")
        clear_node_data(node_1)
        clear_node_data(node_2)
        clear_researcher_data(researcher)

    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(5)

    request.addfinalizer(clear)

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################



def test_01_mnist_pytorch_basic_experiment_run():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 1
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 100,
        'dry_run': False,

    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,)

    exp.run()

    clear_experiment_data(exp)

def test_02_mnist_pytorch_experiment_validation():
    """Test but with more advanced configuration"""

    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 2
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 1,
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
        node_selection_strategy=None,
                tensorboard=True,
        save_breakpoints=True)
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)

    exp.run()
    clear_experiment_data(exp)


def test_03_mnist_pytorch_experiment_scaffold():
    """Test but with more advanced configuration & Scaffold"""

    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 2
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 1,
        'dry_run': False,
        'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=Scaffold(),
        node_selection_strategy=None,
                tensorboard=True,
        save_breakpoints=True)
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)

    exp.run()
    clear_experiment_data(exp)

