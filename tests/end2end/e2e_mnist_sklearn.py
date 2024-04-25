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

from experiments.training_plans.sklearn_mnist_training_plan import SkLearnClassifierTrainingPlan as MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(request):
    """Setup fixture for the module"""

    print("Creating components ---------------------------------------------")
    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")

    researcher = create_component(ComponentType.RESEARCHER, config_name="res.ini")
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



def test_experiment_run_01():
    """Tests running training mnist with basic configuration"""
    model_args = {'n_features': 28*28,
                  'n_classes' : 10,
                  'eta0':1e-6,
                  'random_state':1234,
                  'alpha':0.1 }

    training_args = {
        'epochs': 3, 
        'batch_maxnum': 20,  # can be used to debugging to limit the number of batches per epoch
    #    'log_interval': 1,  # output a logging message every log_interval batches
        'loader_args': {
            'batch_size': 4,
        },
    }
    tags = ['#MNIST', '#dataset']
    rounds = 3

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

