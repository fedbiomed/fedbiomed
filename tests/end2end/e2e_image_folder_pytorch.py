import os
import time

import pytest
from experiments.training_plans.mednist_pytorch_training_plan import (
    ImageFolderTrainingPlan,
    MedNistTrainingPlan,
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

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    print(f"USING PORT {port} for researcher server")
    print("Creating components ---------------------------------------------")

    node_1 = create_node(port=port)
    node_2 = create_node(port=port)
    researcher = create_researcher(port=port)

    mednist_path = get_data_folder("MedNIST-e2e-test")
    dataset1 = {
        "name": "MedNIST",
        "description": "MedNIST DATASET",
        "tags": "#MedNIST,#dataset",
        "data_type": "mednist",
        "path": mednist_path,
    }

    # Folder MedNIST inside root path of MedNistDataset
    # matches the expected structure for ImageFolder dataset
    image_folder_path = os.path.join(mednist_path, "MedNIST")
    dataset2 = {
        "name": "ImageFolder",
        "description": "ImageFolder DATASET",
        "tags": "#imagefolder,#dataset",
        "data_type": "images",
        "path": image_folder_path,
    }

    print("Adding first dataset on first node ----------------------------------------")
    add_dataset_to_node(node_1, dataset1)
    print("Adding first dataset on second node ---------------------------------------")
    add_dataset_to_node(node_2, dataset1)

    time.sleep(1)

    print("Adding second dataset on first node ---------------------------------------")
    add_dataset_to_node(node_1, dataset2)
    print("Adding second dataset on second node --------------------------------------")
    add_dataset_to_node(node_2, dataset2)

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


@pytest.mark.parametrize(
    "tags,training_plan_class,dataset_name",
    [
        (["#MedNIST", "#dataset"], MedNistTrainingPlan, "MedNIST"),
        (["#imagefolder", "#dataset"], ImageFolderTrainingPlan, "ImageFolder"),
    ],
)
def test_pytorch_image_experiment_run(tags, training_plan_class, dataset_name):
    """Tests running training with basic configuration for different datasets"""
    rounds = 1
    training_args = {
        "loader_args": {
            "batch_size": 32,
        },
        "random_seed": 42,
        "optimizer_args": {"lr": 1e-3},
        "epochs": 1,
        "dry_run": False,
        "batch_maxnum": 100,
    }

    model_args = {
        "num_classes": 6,
    }

    print(f"Running experiment with {dataset_name} dataset using tags: {tags}")

    exp = Experiment(
        tags=tags,
        training_plan_class=training_plan_class,
        model_args=model_args,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
    )

    exp.run()

    clear_experiment_data(exp)
