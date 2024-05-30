import pandas as pd
import time
import tqdm
import os
import pytest
import requests
import shutil
from sklearn.model_selection import train_test_split
from zipfile import ZipFile


from helpers import (
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_researcher_data,
    clear_experiment_data,
    get_data_folder,
    create_researcher,
    create_multiple_nodes
)

from experiments.training_plans.ixi_brain_segmentation import UNetTrainingPlan

from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm.tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def download_and_extract_ixi_sample(root_folder):
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip'
    zip_filename = os.path.join(root_folder, '7kd5wj7v7p-3.zip')
    data_folder = os.path.join(root_folder)
    extracted_folder = os.path.join(data_folder, '7kd5wj7v7p-3', 'IXI_sample')

    # Extract if ZIP exists but not folder
    if not os.path.exists(zip_filename):
        # Download if it does not exist
        download_file(url, zip_filename)

    # Check if extracted folder exists
    if os.path.isdir(extracted_folder):
        print(f'Dataset folder already exists in {extracted_folder}')
        return extracted_folder

    with ZipFile(zip_filename, 'r') as zip_obj:
        zip_obj.extractall(data_folder)

    assert os.path.isdir(extracted_folder)
    return extracted_folder


def prepare_ixi_dataset(root_folder: str):
    # Centralized dataset
    centralized_data_folder = download_and_extract_ixi_sample(root_folder)
    csv_global = os.path.join(centralized_data_folder, 'participants.csv')
    allcenters = pd.read_csv(csv_global)

    # Federated Dataset
    federated_data_folder = os.path.join(root_folder, 'Hospital-Centers')

    # Split centers
    center_names = ['Guys', 'HH', 'IOP']
    center_dfs = list()

    for center_name in center_names:
        df = allcenters[allcenters.SITE_NAME == center_name]
        center_dfs.append(df)

        train, test = train_test_split(df, test_size=0.1, random_state=21)

        train_folder = os.path.join(federated_data_folder, center_name, 'train')
        holdout_folder = os.path.join(federated_data_folder, center_name, 'holdout')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
            for subject_folder in train.FOLDER_NAME.values:
                shutil.copytree(
                    src=os.path.join(centralized_data_folder, subject_folder),
                    dst=os.path.join(train_folder, subject_folder),
                    dirs_exist_ok=True
                )
            train_participants_csv = os.path.join(train_folder, 'participants.csv')
            train.to_csv(train_participants_csv)

        if not os.path.exists(holdout_folder):
            os.makedirs(holdout_folder)
            for subject_folder in test.FOLDER_NAME.values:
                shutil.copytree(
                    src=os.path.join(centralized_data_folder, subject_folder),
                    dst=os.path.join(holdout_folder, subject_folder),
                    dirs_exist_ok=True
                )
            test.to_csv(os.path.join(holdout_folder, 'participants.csv'))

# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(post_session, port, request):
    """Setup fixture for the module"""

    data_folder = get_data_folder('IXI-example')
    prepare_ixi_dataset(root_folder=data_folder)

    print("Creating components ---------------------------------------------")
    with create_multiple_nodes(port, 2) as nodes:
        node_1, node_2 = nodes

        researcher = create_researcher(port=port)

        dataset = {
            "name": "IXI",
            "description": "IXI",
            "tags": "ixi-train",
            "data_type": "medical-folder",
            "path": os.path.join(data_folder, "Hospital-Centers", "Guys", "train"),
            "dataset_parameters": {
                "tabular_file": os.path.join(
                    data_folder, "Hospital-Centers", "Guys", "train", "participants.csv"),
                "index_col": 14
            }
        }

        print("Adding first dataset --------------------------------------------")
        add_dataset_to_node(node_1, dataset)
        print("adding second dataset")
        dataset.update({'path': os.path.join(data_folder, "Hospital-Centers", "HH", "train")})
        dataset["dataset_parameters"].update(
            {"tabular_file": os.path.join(data_folder, "Hospital-Centers", "HH", "train", 'participants.csv')})
        add_dataset_to_node(node_2, dataset)

        # start nodes and give some time to start
        node_processes, _ = start_nodes([node_1, node_2])
        time.sleep(10)


        yield

        kill_subprocesses(node_processes)
        clear_researcher_data(researcher)

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################


def test_experiment_run_01():
    """Tests running training mnist with basic configuration"""
    model_args = {
        "spatial_dims": 3,
        'in_channels': 1,
        'out_channels': 2,
        'channels': (16, 32, 64, 128),
        'strides': (2, 2, 2),
        'num_res_units': 1,
        'norm': 'batch',
    }
    training_args = {
        'loader_args': { 'batch_size': 4, 'shuffle': True },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 30,
    }
    print("Instatiating experiment object -----------------------------------")
    tags = ["ixi-train"]
    rounds = 3
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=UNetTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,)

    exp.run()

    clear_experiment_data(exp)

