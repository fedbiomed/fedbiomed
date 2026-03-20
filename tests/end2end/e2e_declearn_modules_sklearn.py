"""
End-to-end tests for researcher-side declearn optimizer modules with
SkLearnTrainingPlan-based training plans.

This file mirrors the style of ``e2e_sklearn.py`` but focuses on covering the
researcher-side declearn OptiModules that can be used as generic aggregation
optimizers with sklearn declearn training plans.

Notes
-----
- These tests intentionally exclude ScaffoldClientModule and
  ScaffoldServerModule from the generic parameterized coverage, because
  scaffold requires dedicated client/server plumbing and already has dedicated
  e2e coverage elsewhere in the test suite.
- The tested module set is checked against ``list_optim_modules()`` so that the
  file fails loudly if new generic modules are added to the registry.
"""

import time

import pytest
from experiments.training_plans.sklearn import SkLearnClassifierTrainingPlanDeclearn
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

from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.optimizers.declearn import (
    AdaGradModule,
    AdamModule,
    EWMAModule,
    MomentumModule,
    RMSPropModule,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule,
    list_optim_modules,
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


MODEL_ARGS = {
    "n_features": 28 * 28,
    "n_classes": 10,
    "eta0": 1e-6,
    "alpha": 0.1,
}

TRAINING_ARGS = {
    "loader_args": {"batch_size": 48},
    "optimizer_args": {"lr": 1e-3},
    "num_updates": 100,
    "test_batch_size": 128,
    "dry_run": False,
}


# Modules that should work as generic researcher-side declearn modules with
# SkLearnClassifierTrainingPlanDeclearn.
TESTED_MODULE_SPECS = {
    "adagrad": [AdaGradModule],
    "adam": [AdamModule],
    "ewma": [EWMAModule],
    "momentum": [MomentumModule],
    "rmsprop": [RMSPropModule],
    "yogi": [YogiModule],
    "yogi-momentum": [YogiMomentumModule],
    "momentum+adam": [MomentumModule, AdamModule],
    "ewma+rmsprop": [EWMAModule, RMSPropModule],
}

# Scaffold modules are intentionally excluded from generic coverage.
EXCLUDED_MODULE_NAMES = {
    ScaffoldClientModule.name,
    ScaffoldServerModule.name,
}


@pytest.mark.parametrize(
    "module_name,module_factories",
    list(TESTED_MODULE_SPECS.items()),
    ids=list(TESTED_MODULE_SPECS.keys()),
)
def test_sklearn_mnist_classifier_with_declearn_module_on_researcher_side(
    module_name, module_factories
):
    """Run a sklearn declearn MNIST experiment with a researcher-side module stack."""

    module_stack = [module_factory() for module_factory in module_factories]

    exp = Experiment(
        tags=["#MNIST", "#dataset"],
        model_args=MODEL_ARGS,
        training_plan_class=SkLearnClassifierTrainingPlanDeclearn,
        training_args=TRAINING_ARGS,
        round_limit=1,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(lr=0.8, modules=module_stack),
        node_selection_strategy=None,
        save_breakpoints=True,
    )

    try:
        exp.run()
        print("After run")
    except BaseException as exc:
        import traceback

        print(f"exp.run raised: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        clear_experiment_data(exp)


def test_sklearn_declearn_module_registry_is_covered():
    """Ensure all generic registry modules are represented in this file.

    This guards against the test silently falling out of sync with the
    declearn module registry.
    """

    registry_names = set(list_optim_modules().keys())
    expected_generic_names = registry_names - EXCLUDED_MODULE_NAMES
    tested_names = {
        AdaGradModule.name,
        AdamModule.name,
        EWMAModule.name,
        MomentumModule.name,
        RMSPropModule.name,
        YogiModule.name,
        YogiMomentumModule.name,
    }

    assert tested_names == expected_generic_names
