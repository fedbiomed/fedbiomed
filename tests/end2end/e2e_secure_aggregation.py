import copy
import time

import pytest
from experiments.training_plans.mnist_pytorch_training_plan import (
    MnistModelScaffoldDeclearn,
    MyTrainingPlan,
)
from helpers import (
    add_dataset_to_node,
    clear_experiment_data,
    get_data_folder,
)

from fedbiomed.common.exceptions import (
    FedbiomedSecureAggregationError,
    FedbiomedStrategyError,
)
from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import ScaffoldServerModule
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.secagg import (
    SecureAggregation,
)
from fedbiomed.researcher.secagg import (
    SecureAggregationSchemes as SecAggSchemes,
)


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(federation):
    """Setup fixture for the module"""
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    print("Creating components ---------------------------------------------")
    with federation.nodes(2, {"security": {"secure_aggregation": "True"}}) as (
        node_1,
        node_2,
    ):
        print("Adding first dataset --------------------------------------------")
        add_dataset_to_node(node_1, dataset)
        print("Adding second dataset -------------------------------------------")
        add_dataset_to_node(node_2, dataset)

        federation.start((node_1, node_2))
        time.sleep(10)

        yield node_1, node_2, federation.researcher


@pytest.fixture
def extra_node_force_secagg(federation):
    """Fixture to add extra node which forces secagg"""

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    with federation.nodes(
        1,
        {
            "security": {
                "secure_aggregation": "True",
                "force_secure_aggregation": "True",
            },
        },
    ) as (node_3,):
        add_dataset_to_node(node_3, dataset)

        federation.start((node_3,))

        # Give some time to researcher
        time.sleep(15)

        yield


@pytest.fixture
def extra_node_no_validation(federation):
    """Fixture to add extra node which disables validation"""

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    with federation.nodes(
        1,
        {
            "security": {
                "secure_aggregation": "True",
                "secagg_insecure_validation": "False",
            },
        },
    ) as (node_3,):
        add_dataset_to_node(node_3, dataset)

        federation.start((node_3,))

        # Give some time to researcher
        time.sleep(15)

        yield


@pytest.fixture
def extra_nodes_for_lom(federation):
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    with federation.nodes(
        3,
        {
            "security": {
                "secure_aggregation": "True",
                "force_secure_aggregation": "True",
            },
        },
    ) as nodes:
        for node in nodes:
            add_dataset_to_node(node, dataset)

        # start nodes and give some time to start
        federation.start(nodes)
        time.sleep(15)

        yield


@pytest.fixture
def extra_nodes_for_lom_8_nodes(federation):
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    with federation.nodes(6, {"security": {"secure_aggregation": "True"}}) as nodes:
        for node in nodes:
            add_dataset_to_node(node, dataset)

        # start nodes and give some time to start
        federation.start(nodes)
        time.sleep(15)

        yield


#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

model_args = {}
tags = ["#MNIST", "#dataset"]
rounds = 2
training_args = {
    "loader_args": {
        "batch_size": 48,
    },
    "optimizer_args": {"lr": 1e-3},
    "num_updates": 100,
    "dry_run": False,
}


def test_01_secagg_joye_libert_pytorch_experiment_basic():
    """Tests running training mnist with basic configuration"""
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.JOYE_LIBERT),
    )

    exp.run()
    clear_experiment_data(exp)


def test_02_secagg_joye_libert_pytorch_breakpoint(setup):
    """Tests running experiment with breakpoint and loading it while secagg active"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.JOYE_LIBERT),
        save_breakpoints=True,
    )

    exp.run()

    # Delete experiment but do not clear its data
    del exp

    # Load experiment from latest breakpoint and continue training
    loaded_exp = Experiment.load_breakpoint()
    print("Running training round after loading the params")
    loaded_exp.run(rounds=2, increase=True)

    # Clear
    clear_experiment_data(loaded_exp)


def test_03_secagg_pytorch_force_secagg(extra_node_force_secagg):
    """Tests failure scenario whereas a node requires secure aggregation
    and researcher does not set it true
    """
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=3,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=False,
        save_breakpoints=True,
    )

    # This should raise exception with default strategy
    with pytest.raises(FedbiomedStrategyError):
        exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_04_secagg_pytorch_no_validation(extra_node_no_validation):
    """Tests failure scenario whereas a researcher requires secure aggregation
    insecure validation and one node refuses to do it
    """
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=3,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=True,
    )

    # This should raise exception with default strategy
    with pytest.raises(FedbiomedSecureAggregationError):
        exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_05_secagg_pytorch_lom():
    """Normal secagg using LOM"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
        save_breakpoints=True,
    )
    exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_06_secagg_lom_pytorch_breakpoint(extra_nodes_for_lom):
    """Tests running experiment with breakpoint and loading it while secagg active LOM"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
        save_breakpoints=True,
    )

    exp.run()
    secagg_id_before = exp.secagg.dh.secagg_id

    # Delete experiment but do not clear its data
    del exp

    # Load experiment from latest breakpoint
    loaded_exp = Experiment.load_breakpoint()

    # Check that `secagg_id` match (good hint that secagg context was properly reloaded)
    print("\nAsserting secagg context match after loading the params")
    secagg_id_after = loaded_exp.secagg.dh.secagg_id
    assert secagg_id_before, secagg_id_after

    # Continue training
    print("\nRunning training round after loading the params")
    loaded_exp.run(rounds=2, increase=True)

    # Clear
    clear_experiment_data(loaded_exp)


def test_07_secagg_pytorch_lom_8_nodes(extra_nodes_for_lom_8_nodes):
    """Secagg using LOM with 8 nodes, which raised some bugs regarding
    failure tests and values conversion
    """

    training_args_8 = copy.deepcopy(training_args)
    training_args_8["dry_run"] = True

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args_8,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
    )
    exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_08_mnist_pytorch_experiment_declearn_scaffold_jls():
    """Test declearn Scaffold optimizer with Joye-Libert secure aggregation"""
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
        secagg=SecureAggregation(scheme=SecAggSchemes.JOYE_LIBERT),
        save_breakpoints=True,
    )
    fed_opt = Optimizer(lr=0.8, modules=[ScaffoldServerModule()])
    exp.set_agg_optimizer(fed_opt)

    exp.run()
    clear_experiment_data(exp)


def test_09_mnist_pytorch_experiment_declearn_scaffold_lom():
    """Test declearn Scaffold optimizer with LOM secure aggregation"""
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
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
        save_breakpoints=True,
    )
    fed_opt = Optimizer(lr=0.8, modules=[ScaffoldServerModule()])
    exp.set_agg_optimizer(fed_opt)

    exp.run()
    clear_experiment_data(exp)
