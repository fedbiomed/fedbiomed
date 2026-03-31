import time

import pytest
from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan
from helpers import (
    add_dataset_to_node,
    clear_component_data,
    clear_experiment_data,
    create_node,
    create_researcher,
    get_data_folder,
    kill_subprocesses,
    start_nodes,
)

from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import (
    AdaGradModule,
    AdamModule,
    EWMAModule,
    MomentumModule,
    RMSPropModule,
    YogiModule,
    YogiMomentumModule,
)
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module."""

    print(f"USING PORT {port} for researcher server")
    print("Creating components ---------------------------------------------")

    node_1 = create_node(port=port)
    node_2 = create_node(port=port)
    researcher = create_researcher(port=port)

    path_data = get_data_folder("MNIST-e2e-test")
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": path_data,
    }

    print("Adding first dataset --------------------------------------------")
    add_dataset_to_node(node_1, dataset)
    print("Adding second dataset -------------------------------------------")
    add_dataset_to_node(node_2, dataset)

    time.sleep(1)

    node_processes, thread = start_nodes([node_1, node_2])

    def clear():
        kill_subprocesses(node_processes)
        thread.join()

        print("Clearing component data")
        clear_component_data(node_1)
        clear_component_data(node_2)
        clear_component_data(researcher)

    print("Sleep 10 seconds. Giving some time for nodes to start")
    time.sleep(10)

    request.addfinalizer(clear)


def _mnist_training_args():
    return {
        "loader_args": {"batch_size": 48},
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "test_batch_size": 128,
        "dry_run": False,
    }


@pytest.mark.parametrize(
    "module_factory,module_name",
    [
        (AdaGradModule, "adagrad"),
        (AdamModule, "adam"),
        (EWMAModule, "ewma"),
        (MomentumModule, "momentum"),
        (RMSPropModule, "rmsprop"),
        (YogiModule, "yogi"),
        (YogiMomentumModule, "yogi-momentum"),
    ],
)
def test_mnist_pytorch_experiment_with_declearn_module(module_factory, module_name):
    """Test a single declearn aggregation module on researcher side."""

    exp = Experiment(
        tags=["#MNIST", "#dataset"],
        model_args={},
        training_plan_class=MyTrainingPlan,
        training_args=_mnist_training_args(),
        round_limit=2,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(lr=0.8, modules=[module_factory()]),
        node_selection_strategy=None,
        tensorboard=True,
        save_breakpoints=True,
    )
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)
    exp.set_test_metric(MetricTypes.ACCURACY)

    try:
        exp.run()
    except Exception as exc:
        raise AssertionError(
            f"E2E declearn module test failed for module '{module_name}': {exc}"
        ) from exc
    finally:
        clear_experiment_data(exp)


@pytest.mark.parametrize(
    "modules,module_names",
    [
        ([MomentumModule(), AdamModule()], "momentum+adam"),
        ([EWMAModule(), RMSPropModule()], "ewma+rmsprop"),
    ],
)
def test_mnist_pytorch_experiment_with_stacked_declearn_modules(modules, module_names):
    """Test stacked declearn aggregation modules on researcher side."""

    exp = Experiment(
        tags=["#MNIST", "#dataset"],
        model_args={},
        training_plan_class=MyTrainingPlan,
        training_args=_mnist_training_args(),
        round_limit=2,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(lr=0.8, modules=modules),
        node_selection_strategy=None,
    )
    exp.set_test_ratio(0.1)
    exp.set_test_on_local_updates(True)
    exp.set_test_on_global_updates(True)
    exp.set_test_metric(MetricTypes.ACCURACY)

    try:
        exp.run()
    except Exception as exc:
        raise AssertionError(
            f"E2E stacked declearn module test failed for '{module_names}': {exc}"
        ) from exc
    finally:
        clear_experiment_data(exp)
