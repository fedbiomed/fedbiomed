"""
End-to-end tests for researcher-side declearn optimizer components with
SkLearnTrainingPlan-based training plans.

This file mirrors the style of ``e2e_sklearn.py`` but focuses on covering the
researcher-side declearn OptiModules and Regularizers that can be used as
generic aggregation optimizers with sklearn declearn training plans.

Notes
-----
- The tested module and regularizer sets are checked against
    ``list_optim_modules()`` and ``list_optim_regularizers()`` so that the file
    fails loudly if new declearn optimizer components are added to the registry.
"""

import os
import time

import pytest
from experiments.training_plans.sklearn import (
    SGDRegressorTrainingPlanDeclearn,
    SGDRegressorTrainingPlanDeclearnScaffold,
    SkLearnClassifierTrainingPlanDeclearn,
)
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
    FedProxRegularizer,
    LassoRegularizer,
    MomentumModule,
    RidgeRegularizer,
    RMSPropModule,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule,
    list_optim_modules,
    list_optim_regularizers,
)
from fedbiomed.common.utils import SHARE_DIR
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

    adni_data_path = os.path.join(
        SHARE_DIR, "notebooks", "data", "CSV", "pseudo_adni_mod.csv"
    )
    adni_dataset = {
        "name": "Adni dataset",
        "description": "Adni DATASET",
        "tags": "#adni",
        "data_type": "csv",
        "path": adni_data_path,
    }

    print("Adding first ADNI dataset --------------------------------------")
    add_dataset_to_node(node_1, adni_dataset)
    print("Adding second ADNI dataset -------------------------------------")
    add_dataset_to_node(node_2, adni_dataset)

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

REGRESSOR_MODEL_ARGS = {
    "max_iter": 2000,
    "tol": 1e-5,
    "eta0": 0.05,
    "n_features": 6,
}

TRAINING_ARGS = {
    "loader_args": {"batch_size": 48},
    "optimizer_args": {"lr": 1e-3},
    "num_updates": 100,
    "test_batch_size": 128,
    "dry_run": False,
}

REGRESSOR_TRAINING_ARGS = {
    "epochs": 5,
    "loader_args": {"batch_size": 32},
    "optimizer_args": {"lr": 0.001},
    "test_ratio": 0.3,
    "test_metric": "MEAN_SQUARE_ERROR",
    "test_on_local_updates": True,
    "test_on_global_updates": True,
}


SCENARIOS = [
    {
        "name": "mnist-adagrad-fedprox",
        "training_plan_class": SkLearnClassifierTrainingPlanDeclearn,
        "tags": ["#MNIST", "#dataset"],
        "model_args": MODEL_ARGS,
        "training_args": TRAINING_ARGS,
        "module_factories": [AdaGradModule],
        "regularizer_factories": [FedProxRegularizer],
    },
    {
        "name": "mnist-adam-lasso",
        "training_plan_class": SkLearnClassifierTrainingPlanDeclearn,
        "tags": ["#MNIST", "#dataset"],
        "model_args": MODEL_ARGS,
        "training_args": TRAINING_ARGS,
        "module_factories": [AdamModule],
        "regularizer_factories": [LassoRegularizer],
    },
    {
        "name": "mnist-ewma-ridge",
        "training_plan_class": SkLearnClassifierTrainingPlanDeclearn,
        "tags": ["#MNIST", "#dataset"],
        "model_args": MODEL_ARGS,
        "training_args": TRAINING_ARGS,
        "module_factories": [EWMAModule],
        "regularizer_factories": [RidgeRegularizer],
    },
    {
        "name": "mnist-yogi-momentum-fedprox",
        "training_plan_class": SkLearnClassifierTrainingPlanDeclearn,
        "tags": ["#MNIST", "#dataset"],
        "model_args": MODEL_ARGS,
        "training_args": TRAINING_ARGS,
        "module_factories": [YogiMomentumModule],
        "regularizer_factories": [FedProxRegularizer],
    },
    {
        "name": "mnist-momentum-adam-lasso-ridge",
        "training_plan_class": SkLearnClassifierTrainingPlanDeclearn,
        "tags": ["#MNIST", "#dataset"],
        "model_args": MODEL_ARGS,
        "training_args": TRAINING_ARGS,
        "module_factories": [MomentumModule, AdamModule],
        "regularizer_factories": [LassoRegularizer, RidgeRegularizer],
    },
    {
        "name": "adni-momentum-fedprox",
        "training_plan_class": SGDRegressorTrainingPlanDeclearn,
        "tags": ["#adni"],
        "model_args": REGRESSOR_MODEL_ARGS,
        "training_args": REGRESSOR_TRAINING_ARGS,
        "module_factories": [MomentumModule],
        "regularizer_factories": [FedProxRegularizer],
    },
    {
        "name": "adni-rmsprop-lasso",
        "training_plan_class": SGDRegressorTrainingPlanDeclearn,
        "tags": ["#adni"],
        "model_args": REGRESSOR_MODEL_ARGS,
        "training_args": REGRESSOR_TRAINING_ARGS,
        "module_factories": [RMSPropModule],
        "regularizer_factories": [LassoRegularizer],
    },
    {
        "name": "adni-yogi-ridge",
        "training_plan_class": SGDRegressorTrainingPlanDeclearn,
        "tags": ["#adni"],
        "model_args": REGRESSOR_MODEL_ARGS,
        "training_args": REGRESSOR_TRAINING_ARGS,
        "module_factories": [YogiModule],
        "regularizer_factories": [RidgeRegularizer],
    },
    {
        "name": "adni-ewma-rmsprop-fedprox",
        "training_plan_class": SGDRegressorTrainingPlanDeclearn,
        "tags": ["#adni"],
        "model_args": REGRESSOR_MODEL_ARGS,
        "training_args": REGRESSOR_TRAINING_ARGS,
        "module_factories": [EWMAModule, RMSPropModule],
        "regularizer_factories": [FedProxRegularizer],
    },
    {
        "name": "adni-scaffold-server",
        "training_plan_class": SGDRegressorTrainingPlanDeclearnScaffold,
        "tags": ["#adni"],
        "model_args": REGRESSOR_MODEL_ARGS,
        "training_args": REGRESSOR_TRAINING_ARGS,
        "module_factories": [ScaffoldServerModule],
        "regularizer_factories": [],
    },
]


TESTED_MODULE_NAMES = {
    AdaGradModule.name,
    AdamModule.name,
    EWMAModule.name,
    MomentumModule.name,
    RMSPropModule.name,
    ScaffoldClientModule.name,
    ScaffoldServerModule.name,
    YogiModule.name,
    YogiMomentumModule.name,
}

TESTED_REGULARIZER_NAMES = {
    FedProxRegularizer.name,
    LassoRegularizer.name,
    RidgeRegularizer.name,
}


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[scenario["name"] for scenario in SCENARIOS],
)
def test_sklearn_training_plans_with_declearn_optimizer_components(scenario):
    """Run sklearn declearn experiments with mixed researcher-side optimizer stacks."""

    module_stack = [module_factory() for module_factory in scenario["module_factories"]]
    regularizer_stack = [
        regularizer_factory()
        for regularizer_factory in scenario["regularizer_factories"]
    ]

    exp = Experiment(
        tags=scenario["tags"],
        model_args=scenario["model_args"],
        training_plan_class=scenario["training_plan_class"],
        training_args=dict(scenario["training_args"]),
        round_limit=1,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(
            lr=0.8,
            modules=module_stack,
            regularizers=regularizer_stack,
        ),
        node_selection_strategy=None,
        save_breakpoints=True,
    )

    try:
        exp.run()
        print(f"After run for scenario '{scenario['name']}'")
    except BaseException as exc:
        import traceback

        print(
            f"exp.run raised for scenario '{scenario['name']}': "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        clear_experiment_data(exp)


def test_sklearn_declearn_module_registry_is_covered():
    """Ensure all registry modules are represented in this file."""

    registry_names = set(list_optim_modules().keys())
    assert TESTED_MODULE_NAMES == registry_names


def test_sklearn_declearn_regularizer_registry_is_covered():
    """Ensure all generic regularizers are represented in this file."""

    registry_names = set(list_optim_regularizers().keys())
    assert TESTED_REGULARIZER_NAMES == registry_names
