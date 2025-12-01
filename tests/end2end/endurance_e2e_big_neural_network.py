"""
Test module to test a big model (60MB) using dry run training
"""

import gc
import os
import sys

import psutil
import pytest
import torch
from experiments.training_plans.mnist_pytorch_training_plan import (
    BigModelMyTrainingPlan,
)
from helpers import (
    add_dataset_to_node,
    clear_component_data,
    clear_experiment_data,
    create_multiple_nodes,
    create_researcher,
    get_data_folder,
    kill_subprocesses,
    start_nodes,
)

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold
from fedbiomed.researcher.federated_workflows import Experiment


def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""
    data_folder = get_data_folder("MNIST-e2e-test")
    with create_multiple_nodes(port, 3) as nodes:
        dataset = {
            "name": "MNIST",
            "description": "MNIST DATASET",
            "tags": "#MNIST,#dataset",
            "data_type": "default",
            "path": data_folder,
        }

        for node in nodes:
            add_dataset_to_node(node, dataset)

        researcher = create_researcher(port=port)
        node_processes, thread = start_nodes(list(nodes))

        yield

        kill_subprocesses(node_processes)
        thread.join()
        print("Clearing component data")
        clear_component_data(researcher)


#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################


def test_01_mnist_pytorch_big_model_training_dry_run():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ["#MNIST", "#dataset"]
    rounds = 30
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "dry_run": True,
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=BigModelMyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        retain_full_history=False,
    )

    print("Memory before Experiment:", memory_mb(), "MB")
    exp.run()
    print("Memory after Experiment:", memory_mb(), "MB")

    for name, value in exp.__dict__.items():
        print(name, type(value), sys.getsizeof(value))

    clear_experiment_data(exp)

    # Clear known heavy attributes if they exist
    for attr in [
        "_training_plan",
        "_model_training_plan",
        "model",
        "training_plan",
        "aggregator",
        "training_args",
        "model_args",
    ]:
        if hasattr(exp, attr):
            try:
                delattr(exp, attr)
            except AttributeError:
                pass

    del exp

    # PyTorch-specific memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    print("Memory after cleanup:", memory_mb(), "MB")


def test_02_mnist_pytorch_big_model_training_dry_run_native_scaffold():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ["#MNIST", "#dataset"]
    rounds = 30
    training_args = {
        "loader_args": {
            "batch_size": 48,
        },
        "optimizer_args": {"lr": 1e-3},
        "num_updates": 100,
        "dry_run": True,
    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=BigModelMyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=Scaffold(),
        node_selection_strategy=None,
        retain_full_history=False,
    )

    print("Memory before Experiment:", memory_mb(), "MB")
    exp.run()
    print("Memory after Experiment:", memory_mb(), "MB")

    for name, value in exp.__dict__.items():
        print(name, type(value), sys.getsizeof(value))

    clear_experiment_data(exp)

    # Clear known heavy attributes if they exist
    for attr in [
        "_training_plan",
        "_model_training_plan",
        "model",
        "training_plan",
        "aggregator",
        "training_args",
        "model_args",
    ]:
        if hasattr(exp, attr):
            try:
                delattr(exp, attr)
            except AttributeError:
                pass

    del exp

    # PyTorch-specific memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    print("Memory after cleanup:", memory_mb(), "MB")
