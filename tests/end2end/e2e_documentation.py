from contextlib import contextmanager
import copy
import os
import tempfile
import time
import pytest
import urllib
import zipfile

import torch

from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    execute_script,
    clear_node_data,
    create_multiple_nodes,
    create_researcher,
    get_data_folder,
    create_node,
    clear_researcher_data,
    configure_secagg,
    secagg_certificate_registration
)


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.researcher.environ import environ
from nbclient.exceptions import CellExecutionError


mnist_dataset = {
    "name": "MNIST",
    "description": "MNIST DATASET",
    "tags": "#MNIST,#dataset",
    "data_type": "default",
    "path": get_data_folder('')}

mednist_data_path = get_data_folder('MedNist_e2e')
mednist_dataset = {
        "name": "Mednist part 1",
        "description": "Mednist part 1",
        "tags": "mednist,#MEDNIST,#dataset",
        "data_type": "mednist",
        "path": mednist_data_path
    }


# Set up nodes and start
@pytest.fixture(scope='module', autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    print("Creating components ---------------------------------------------")
    os.environ['RESEARCHER_SERVER_PORT'] = port

    with create_multiple_nodes(port, 2) as nodes:
        node_1, node_2 = nodes

        researcher = create_researcher(port)

        for d in (mnist_dataset, mednist_dataset):
            add_dataset_to_node(node_1, d)
            add_dataset_to_node(node_2, d)


        # Starts the nodes
        node_processes, _ = start_nodes([node_1, node_2])


        yield node_1, node_2, researcher


        kill_subprocesses(node_processes)
        clear_researcher_data(researcher)


def remove_data(path: str):
    """Remove some generated data during tests"""
    # here we removed model file created during test
    model_file_path = os.path.join(path, 'trained_model')
    if os.path.isfile(model_file_path):
        os.remove(model_file_path)

# @contextmanager
# def remove_symbolic_link(sym_link: str):
#     yield
#     os.unlink(sym_link)

@pytest.fixture
def extra_node(port, dataset):
    """Fixture to add extra node"""

    node_3 = create_node(port)

    # Starts the nodes
    node_processes, _ = start_nodes([node_3])
    add_dataset_to_node(node_3, dataset)

    yield

    kill_subprocesses(node_processes)
    clear_node_data(node_3)


def test_documentation_01_pytorch_mnist_basic_example():
    """Tests mnist basic example"""

    execute_script(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'pytorch',
                                 '01_PyTorch_MNIST_Single_Node_Tutorial.ipynb'
    ))
    remove_data(os.path.join(ROOT_DIR, 'docs', 'tutorials', 'pytorch'))


def test_documentation_02_create_your_custom_training_plan(setup):
    """Tests"""

    celeba_folder = get_data_folder('Celeba')
    celeba_preprocessed = os.path.join(celeba_folder, 'celeba_preprocessed')
    celeba_raw = os.path.join(celeba_folder, 'Celeba_raw')

    if not os.path.isdir(celeba_preprocessed):
        pytest.skip(
            'Celeba raw dataset is not existing please follow tutorial and '
            f'put the dataset in the folder {celeba_folder}'
        )

    if not os.path.isdir(celeba_raw):
        pytest.skip(
            'Celeba raw dataset is not existing please follow tutorial and '
            f'put the dataset in the folder {celeba_folder}'
       )

    parent_dir = os.path.join(environ["ROOT_DIR"], "notebooks", "data", "Celeba")
    # FIXME: what does the first symlink do exactly?
    # FIXME: create context manager /decorator to remove symbolic link in case of failure/error raised
    # os.symlink(
    #     os.path.join(celeba_folder, 'Celaba_raw'),
    #     os.path.join(parent_dir, 'Celeba_raw'))
    os.symlink(
        os.path.join(celeba_folder, 'celeba_preprocessed'),
        os.path.join(parent_dir, 'celeba_preprocessed'))


    # TODO: copy or skip test if dataset is not found
    node_1, node_2, _ = setup

    celeba_dataset_n1 = {
        "name": "celeba",
        "description": "celeba DATASET",
        "tags": "#celeba,#dataset",
        "data_type": "images",
        "path": os.path.join(celeba_folder, 'celeba_preprocessed', 'data_node_1')
    }
    celeba_dataset_n2 = copy.deepcopy(celeba_dataset_n1)
    celeba_dataset_n2 = {
        **celeba_dataset_n1,
        'path': os.path.join(celeba_folder, 'celeba_preprocessed', 'data_node_2')}

    add_dataset_to_node(node_1, celeba_dataset_n1)
    add_dataset_to_node(node_2, celeba_dataset_n2)

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'pytorch',
                                '02_Create_Your_Custom_Training_Plan.ipynb'
    ))


    # remove symbolic links once test has been passed 
    os.unlink(os.path.join(parent_dir, 'Celeba_raw'))


    remove_data(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'pytorch',))


def test_documentation_03_pytroch_used_cars_dataset_example(setup):
    """Runs UsedCars tutorial"""

    data_folder = get_data_folder('UsedCars')
    # FIXME: better name the path notebooks_data
    notebooks_data = os.path.join(ROOT_DIR, 'notebooks', 'data', 'UsedCars')

    #os.makedirs(notebooks_data, exist_ok=True)
    #os.makedirs(os.path.join(notebooks_data, 'raw'))
    # if os.path.isdir(os.path.join(data_folder, 'raw') ):
    #     pytest.skip(f"Pleas follow the tutorials and privode raw and processed dataset in the {data_folder} directory")

    
    os.symlink(
        os.path.join(data_folder),
        os.path.join(notebooks_data))


    dataset_node_1 = os.path.join(data_folder, 'audi_transformed.csv')
    dataset_node_2 = os.path.join(data_folder, 'bmw_transformed.csv')
    dataset_node_3 = os.path.join(data_folder, 'ford_transformed.csv')
    # TODO: should we copy dataset from ROOT_DIR/data -> ROOT_DIR/notebooks/data/UsedCars ? instead
    # of skipping test
    #_parent_folder_path = get_data_folder('UsedCars')
    print("DEBUG", data_folder, dataset_node_1)
    if not all(os.path.isfile(i) for i in (dataset_node_1, dataset_node_2, dataset_node_3)):
        os.unlink(notebooks_data)
        pytest.skip('Data files for UsedCars example is not existing '
            f'Please see the tutorial and create data files raw and processed in {data_folder}')

    used_cars_1 = {
        "name": "used cars",
        "description": "Used Cars DATASET",
        "tags": "#UsedCars",
        "data_type": "csv",
        "path": dataset_node_1
    }

    used_cars_2 = {
        **used_cars_1, 'path': dataset_node_2}


    node_1, node_2, _ = setup

    add_dataset_to_node(node_1, used_cars_1)
    add_dataset_to_node(node_2, used_cars_2)

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'pytorch',
                                '03_PyTorch_Used_Cars_Dataset_Example.ipynb'
    ))

    os.unlink(notebooks_data)
    remove_data(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'pytorch',))


# Tests for scikit-learn
def test_documentation_01_sklearn_mnist_classification_tutorial():
    """Sklearn MNIST classification tutorial"""

    execute_script(os.path.join(
        environ['ROOT_DIR'],
        'docs',
        'tutorials',
        'scikit-learn',
        '01_sklearn_MNIST_classification_tutorial.ipynb'))

    # NOTA: MNIST test dataset should be removed since it has been loaded in a temporary folder
    remove_data(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'scikit-learn',))

def test_documentation_02_sklearn_sgd_regression_tutorial(setup):
    """SGD Regression"""

    node_1, node_2, _ = setup

    data_path = os.path.join('notebooks', 'data', 'CSV', 'pseudo_adni_mod.csv')

    pseudo_adni_dataset = {
        "name": "ADNI_dataset",
        "description": "ADNI DATASET",
        "tags": "adni",
        "data_type": "csv",
        "path": data_path
    }

    add_dataset_to_node(node_1, pseudo_adni_dataset)
    add_dataset_to_node(node_2, pseudo_adni_dataset)

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'scikit-learn',
                                '02_sklearn_sgd_regressor_tutorial.ipynb'))

    remove_data(os.path.join(environ['ROOT_DIR'],
                                 'docs',
                                 'tutorials',
                                 'scikit-learn',))

def test_documentation_01_fedopt_and_scaffold(setup, provide_mednist_dataset):
    # TODO: download dataset when not available
    node_1, node_2, researcher = setup

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'optimizers',
                                '01-fedopt-and-scaffold.ipynb'))
    remove_data(
        os.path.join(environ['ROOT_DIR'],
                     'docs',
                     'tutorials',
                     'optimizers'))

######################
# Advanced tutorials
def test_documentation_01_in_depth_experiment_configuration(setup):
    """Tests general in depth experiment configuration notebook"""

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'advanced',
                                'in-depth-experiment-configuration.ipynb'))

    remove_data(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'advanced',))

def test_documentation_02_training_with_gpu():
    researcher = create_component(ComponentType.RESEARCHER, config_name="config_researcher.ini")
    node_1 = create_component(ComponentType.NODE, config_name="config1.ini")
    node_2 = create_component(ComponentType.NODE, config_name="config2.ini")
    node_3 = create_component(ComponentType.NODE, config_name="config3.ini")

    time.sleep(1)

    # Starts the nodes
    # node_1 with `--gpu` statement
    # node_2 with `--gpu-only --gpu-num 1` statements
    # node 3 with no additional statement

    if torch.cuda.is_available():
        # case where GPU is available
        node_processes, _ = start_nodes([node_1, node_2, node_3,],
                                        [['--gpu'], ['--gpu-only', '--gpu-num', '1'], []])

    else:
        # case where GPu is not available
        node_processes, _ = start_nodes([node_1, node_2, node_3,],
                                        [['--gpu'], ['--gpu'], []])
    mnist_dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }

    add_dataset_to_node(node_1, mnist_dataset)
    add_dataset_to_node(node_2, mnist_dataset)
    add_dataset_to_node(node_3, mnist_dataset)
    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'advanced',
                                'training-with-gpu.ipynb'))

    # finish processes and clean
    kill_subprocesses(node_processes)
    time.sleep(5)
    print("Clearing component data")
    clear_node_data(node_1)
    clear_node_data(node_2)
    clear_researcher_data(researcher)

    # TODO: check that node_3 is not used with gpu during training, but node 2 does

    remove_data(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'advanced',))

############################
# MONAI specific notebooks
############################

def test_documentation_01_monai_2d_image_classification(setup, provide_mednist_dataset):
    node_1, node_2, _= setup

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'monai',
                                '01_monai-2d-image-classification.ipynb'))

    remove_data(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'monai',))


def test_documentation_02_monai_2d_image_registration(setup, provide_mednist_dataset):
    node_1, node_2, researcher = setup
    dataset_mednist_1, dataset_mednist_2, _ = provide_mednist_dataset
    add_dataset_to_node(node_1, dataset_mednist_1)
    add_dataset_to_node(node_2, dataset_mednist_2)

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'monai',
                                '02_monai-2d-image-registration.ipynb'))
    remove_data(os.path.join(environ['ROOT_DIR'],
                            'docs',
                            'tutorials',
                            'monai',))



def test_documentation_medical_medical_image_segmentation_unet_library():
    # TODO: some cells in the notebook should not be executed...
    # TODO: update when non-runable cell tags will be implemented
    pass

######################
# security notebooks
#####################

def test_documentation_security_differential_privacy_with_opacus_on_fedbiomed(setup):
    node_1, node_2, researcher = setup
    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'security',
                                'differential-privacy-with-opacus-on-fedbiomed.ipynb'))

    remove_data(os.path.join(environ['ROOT_DIR'],
                            'docs',
                            'tutorials',
                            'security',))



def test_documentation_security_non_private_local_central_dp_monai2d_image_registration(extra_node):
    """Test local and central dp"""

    add_dataset_to_node(extra_node, mednist_dataset)

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'security',
                                'non-private-local-central-dp-monai-2d-image-registration.ipynb'))

    remove_data(os.path.join(environ['ROOT_DIR'],
                             'docs',
                             'tutorials',
                             'security',))


def test_documentation_security_secure_aggregation():
    mnist_dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }
    print("Configure secure aggregation ---------------------------------------------")
    configure_secagg()

    print("Creating§ components ---------------------------------------------")
    node_1 = create_component(
        ComponentType.NODE,
        config_name="config_n1.ini",
        config_sections={'security': {'secure_aggregation': 'True'}})

    node_2 = create_component(
        ComponentType.NODE,
        config_name="config_n2.ini",
        config_sections={'security': {'secure_aggregation': 'True'}})


    print("Creating researcher component ---------------------------------------------")
    researcher = create_component(ComponentType.RESEARCHER, config_name="res.ini")


    print("Register certificates ---------------------------------------------")
    secagg_certificate_registration()

    print("Adding first dataset --------------------------------------------")
    add_dataset_to_node(node_1, mnist_dataset)
    print("adding second dataset")
    add_dataset_to_node(node_2, mnist_dataset)

    time.sleep(1)

    # Starts the nodes
    node_processes, _ = start_nodes([node_1, node_2])

    execute_script(os.path.join(environ['ROOT_DIR'],
                                'docs',
                                'tutorials',
                                'security',
                                'secure-aggregation.ipynb'))

    # TODO:  finish writing the test
    kill_subprocesses(node_processes)

    clear_node_data(node_1)
    clear_node_data(node_2)

    clear_researcher_data(researcher)
    remove_data(os.path.join(environ['ROOT_DIR'],
                             'docs',
                             'tutorials',
                             'security',))


def test_documentation_security_training_with_approved_training_plans():
    # test not working because an exception is raised in the notebook, and
    # in cannot figure out a way to catch exceptions from a notebook cell

    # TODO: complete test
    pass
    # node_1 = create_component(
    #     ComponentType.NODE,
    #     config_name="config_n1.ini",
    #     config_sections={'security': {'training_plan_approval': 'True'}})

    # node_2 = create_component(
    #     ComponentType.NODE,
    #     config_name="config_n2.ini",
    #     config_sections={'security': {'training_plan_approval': 'True'}})
    # researcher = create_component(ComponentType.RESEARCHER, config_name="res.ini")

    # add_dataset_to_node(node_1, dataset)
    # add_dataset_to_node(node_2, dataset)

    # time.sleep(1)

    # nodes_processes, _ = start_nodes([node_1, node_2,])
    # # here we will test the first notebook cells till the cell that raises exception

    # with pytest.raises(CellExecutionError):
    #     execute_script(os.path.join(environ['ROOT_DIR'],
    #                                 'docs',
    #                                 'tutorials',
    #                                 'security',
    #                                 'training-with-approved-training-plans.ipynb'))
    # # TODO: finish writing the test

    # kill_subprocesses(nodes_processes)
    # clear_node_data(node_1)
    # clear_node_data(node_2)
    # clear_researcher_data(researcher)
    # remove_data(os.path.join(environ['ROOT_DIR'],
    #                          'docs',
    #                          'tutorials',
    #                          'security',))
