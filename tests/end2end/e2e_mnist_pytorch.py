import time
import pytest

from helpers import (
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_component_data,
    clear_experiment_data,
    get_data_folder,
    create_node,
    create_researcher,
)

from experiments.training_plans.mnist_pytorch_training_plan import (
    MnistModelScaffoldDeclearn,
    MyTrainingPlan,
)

from fedbiomed.common.metrics import MetricTypes
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold
from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import ScaffoldServerModule


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    print(f"USING PORT {port} for researcher erver")
    print("Creating components ---------------------------------------------")

    node_1 = create_node(port=port)
    node_2 = create_node(port=port)
    researcher = create_researcher(port=port)

    data = get_data_folder("MNIST-e2e-test")
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": data,
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
        clear_component_data(node_1)
        clear_component_data(node_2)
        clear_component_data(researcher)

    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(10)

    request.addfinalizer(clear)


#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################


def test_01_mnist_pytorch_basic_experiment_run():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ["#MNIST", "#dataset"]
    rounds = 1
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "dry_run": False,
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
    )

    exp.run()

    clear_experiment_data(exp)


def test_02_mnist_pytorch_experiment_validation():
    """Test but with more advanced configuration"""

    model_args = {}
    tags = ["#MNIST", "#dataset"]
    rounds = 2
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "test_batch_size": 128,
        "dry_run": False,
        #'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
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
        save_breakpoints=True,
    )
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)

    exp.run()
    clear_experiment_data(exp)


def test_03_mnist_pytorch_experiment_scaffold():
    """Test but with more advanced configuration & Scaffold"""

    model_args = {}
    tags = ["#MNIST", "#dataset"]
    rounds = 2
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "test_batch_size": 128,
        "dry_run": False,
        #'batch_maxnum': 100 # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
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
        save_breakpoints=True,
    )
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)
    exp.set_test_metric(MetricTypes.ACCURACY)
    exp.run()
    clear_experiment_data(exp)


def test_04_mnist_pytorch_experiment_declearn_scaffold():
    model_args = {}
    tags = ["#MNIST", "#dataset"]
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 200,
        "dry_run": False,
    }

    rounds = 5
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MnistModelScaffoldDeclearn,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        tensorboard=True,
        save_breakpoints=True,
    )
    fed_opt = Optimizer(lr=0.8, modules=[ScaffoldServerModule()])
    exp.set_agg_optimizer(fed_opt)
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)
    exp.set_test_metric(MetricTypes.ACCURACY)
    exp.run()
    clear_experiment_data(exp)
