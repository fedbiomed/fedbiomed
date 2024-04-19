import os
import time
import pytest
import urllib
import zipfile
from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    execute_script,
    clear_node_data,
    clear_experiment_data,
    clear_researcher_data)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold
from fedbiomed.researcher.environ import environ


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(request):
    """Setup fixture for the module"""

    print("Creating components ---------------------------------------------")
<<<<<<< HEAD
    
    researcher = create_component(ComponentType.RESEARCHER, config_name="config_researcher.ini")
    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")
    
=======

    researcher = create_component(ComponentType.RESEARCHER, config_name="config_researcher.ini")
    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")

>>>>>>> 4f7d350c (Solves stucked end-to-end test)

    time.sleep(1)

    # Starts the nodes
    node_processes, _ = start_nodes([node_1, ])

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }
    add_dataset_to_node(node_1, dataset)
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

    return node_1, node_2, researcher

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################
def test_documentation_01_pytorch_mnist_basic_example(setup):
    node_1, node_2, researcher = setup
    execute_script(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'pytorch',
                                 '01_PyTorch_MNIST_Single_Node_Tutorial.ipynb'
    ))

    # TODO: add another dataset to node during the execution


def test_documentation_02_create_your_custom_training_plan(setup):
    # FIXME: in this test, we assume that the first cell has already been
    # executed
    #from fedbiomed.researcher.environ import environ
    parent_dir = os.path.join(environ["ROOT_DIR"], "notebooks", "data", "Celeba")
    # download and unzip
    url_celeba = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg"

    # import gdown
    # zip_path = os.path.join(parent_dir, 'img_align_celeba.zip')
    # gdown.download(url_celeba, parent_dir, quiet=False)
    # with zipfile.ZipFile(zip_path, "r") as f:
    #     f.extractall(parent_dir)

    #url_anno_celeba =
    # first cell (cut and pasted)

    # import pandas as pd
    # import shutil

    # TODO: find a way to download celeba data


    # # Celeba folder

    # celeba_raw_folder = os.path.join("Celeba_raw", "raw")
    # img_dir = os.path.join(parent_dir, celeba_raw_folder, 'img_align_celeba') + os.sep
    # out_dir = os.path.join(parent_dir, "celeba_preprocessed")

    # # Read attribute CSV and only load Smilling column
    # df = pd.read_csv(os.path.join(parent_dir, celeba_raw_folder, 'list_attr_celeba.txt'),
    #                 sep="\s+", skiprows=1, usecols=['Smiling'])

    # # data is on the form : 1 if the person is smiling, -1 otherwise. we set all -1 to 0 for the model to train faster
    # df.loc[df['Smiling'] == -1, 'Smiling'] = 0

    # # Split csv in 3 part
    # length = len(df)
    # data_node_1 = df.iloc[:int(length/3)]
    # data_node_2 = df.iloc[int(length/3):int(length/3) * 2]
    # data_node_3 = df.iloc[int(length/3) * 2:]

    # # Create folder for each node
    # if not os.path.exists(os.path.join(out_dir, "data_node_1")):
    #     os.makedirs(os.path.join(out_dir, "data_node_1", "data"))
    # if not os.path.exists(os.path.join(out_dir, "data_node_2")):
    #     os.makedirs(os.path.join(out_dir, "data_node_2", "data"))
    # if not os.path.exists(os.path.join(out_dir, "data_node_3")):
    #     os.makedirs(os.path.join(out_dir, "data_node_3", "data"))

    # # Save each node's target CSV to the corect folder
    # data_node_1.to_csv(os.path.join(out_dir, 'data_node_1', 'target.csv'), sep='\t')
    # data_node_2.to_csv(os.path.join(out_dir, 'data_node_2', 'target.csv'), sep='\t')
    # data_node_3.to_csv(os.path.join(out_dir, 'data_node_3', 'target.csv'), sep='\t')

    # # Copy all images of each node in the correct folder
    # for im in data_node_1.index:
    #     shutil.copy(img_dir+im, os.path.join(out_dir,"data_node_1", "data", im))
    # print("data for node 1 succesfully created")

    # for im in data_node_2.index:
    #     shutil.copy(img_dir+im, os.path.join(out_dir, "data_node_2", "data", im))
    # print("data for node 2 succesfully created")

    # for im in data_node_3.index:
    #     shutil.copy(img_dir+im, os.path.join(out_dir, "data_node_3", "data", im))
    # print("data for node 3 succesfully created")

    # # end of first cell in the notebook
    # node_1, node_2, researcher = setup


def test_documentation_03_pytroch_used_cars_dataset_example(setup):
    # TODO: this require to download the used car dataset example (needs to log into a kaggle account)
    pass

def test_documentation_04_aggregation_in_fedbiomed():
    # TODO: this test require Flamby dataset (usually installed manually)
    pass


# Tests for scikit-learn
def test_documentation_01_sklearn_mnist_classification_tutorial(setup):
    node_1, node_2, researcher = setup

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'scikit-learn',
                                '01_sklearn_MNIST_classification_tutorial.ipynb'
    ))

    # NOTA: MNIST test dataset should be removed since it has been loaded in a temporary folder
