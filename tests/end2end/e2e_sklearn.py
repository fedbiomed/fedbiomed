import os
import time
import pytest

from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_node_data,
    clear_experiment_data,
    clear_researcher_data,
    create_researcher,
    get_data_folder,
    create_multiple_nodes
)

from experiments.training_plans.sklearn import (
    PerceptronTraining,
    SGDRegressorTrainingPlan,
    SGDClassifierTrainingPlan,
    SkLearnClassifierTrainingPlanDeclearn,
    SGDRegressorTrainingPlanDeclearn
)

from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.common.metrics import MetricTypes

from sklearn import datasets
import numpy as np

# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    with create_multiple_nodes(port, 3) as nodes:
        node_1, node_2, node_3 = nodes

        # Create researcher component
        researcher = create_researcher(
            port=port,
            config_name="config_researcher_sklearn.ini",
            config_sections={'server': {'port': port}},
        )

        # Starts the nodes
        node_processes, thread = start_nodes([node_1, node_2, node_3])

        # Generate datasets
        p1, p2, p3 = generate_sklearn_classification_dataset()
        dataset = {
            "name": "MNIST",
            "description": "MNIST DATASET",
            "tags": "#csv-dataset-classification",
            "data_type": "csv"}


        d1 = {**dataset, "path": p1}
        d2 = {**dataset, "path": p2}
        d3 = {**dataset, "path": p3}
        add_dataset_to_node(node_1, d1)
        add_dataset_to_node(node_2, d2)
        add_dataset_to_node(node_3, d3)


        # Add MNIST dataset
        datafolder = get_data_folder('')
        dataset = {
            "name": "MNIST",
            "description": "MNIST DATASET",
            "tags": "#MNIST,#dataset",
            "data_type": "default",
            "path": datafolder
        }
        add_dataset_to_node(node_1, dataset)
        add_dataset_to_node(node_2, dataset)
        add_dataset_to_node(node_3, dataset)


        data_path = os.path.join('notebooks', 'data', 'CSV', 'pseudo_adni_mod.csv')
        dataset = {
            "name": "Adni dataset",
            "description": "Adni DATASET",
            "tags": "#adni",
            "data_type": "csv",
            "path": data_path}

        add_dataset_to_node(node_1, dataset)
        add_dataset_to_node(node_2, dataset)
        add_dataset_to_node(node_3, dataset)

        time.sleep(1)

        # Good to wait 3 second to give time to nodes start
        print("Sleep 5 seconds. Giving some time for nodes to start")
        time.sleep(5)


        # Run tests
        yield node_1, node_2, node_3, researcher


        kill_subprocesses(node_processes)
        thread.join()

        print("Clearing researcher data")
        clear_researcher_data(researcher)



def generate_sklearn_classification_dataset():
    """Generate testing data"""

    path = get_data_folder('sklearn-e2e-tests')
    p1 = os.path.join(path , 'c1.csv')
    p2 = os.path.join(path , 'c2.csv')
    p3 = os.path.join(path , 'c3.csv')

    # If there is path stop generating the data again
    if os.path.isdir(path):
        return p1, p2, p3

    X,y = datasets.make_classification(
        n_samples=300,
        n_features=20,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=123
    )
    C1 = X[:150,:]
    C2 = X[150:250,:]
    C3 = X[250:300,:]

    y1 = y[:150].reshape([150,1])
    y2 = y[150:250].reshape([100,1])
    y3 = y[250:300].reshape([50,1])


    n1 = np.concatenate((C1, y1), axis=1)
    np.savetxt(p1, n1, delimiter=',')

    n2 = np.concatenate((C2, y2), axis=1)
    np.savetxt(p2 ,n2, delimiter=',')

    n3 = np.concatenate((C3, y3), axis=1)
    np.savetxt(p3,n3, delimiter=',')


    return p1, p2, p3


def test_01_sklearn_perceptron():
    """Tests sklearn perceptron"""

    print("In the test")
    n_features = 20
    n_classes = 2

    model_args = {
        'max_iter':1000,
        'tol': 1e-3 ,
        'n_features' : n_features,
        'n_classes' : n_classes
    }

    training_args = {
        'epochs': 5,
        'loader_args': { 'batch_size': 1 }
    }

    tags =  ['#csv-dataset-classification']
    rounds = 2

    # search for corresponding datasets across nodes datasets
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=PerceptronTraining,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)

def test_02_sklean_sgdregressor():
    """Test SGDRegressor using Adni dataset"""

    _seed = 1234
    model_args = {
        'max_iter':2000,
        'tol': 1e-5,
        'eta0':0.05,
        'n_features': 6,
        'random_state': _seed
    }

    training_args = {
        'epochs': 5,
        'loader_args': { 'batch_size': 10, },
        'test_ratio':.3,
        'test_metric': MetricTypes.MEAN_SQUARE_ERROR,
        'test_on_local_updates': True,
        'test_on_global_updates': True
    }

    tags =  ['#adni']
    rounds = 3

    # select nodes participating to this experiment
    exp = Experiment(tags=tags,
                     model_args=model_args,
                     training_plan_class=SGDRegressorTrainingPlan,
                     training_args=training_args,
                     round_limit=rounds,
                     aggregator=FedAverage(),
                     node_selection_strategy=None,
                     save_breakpoints=True)

    exp.run()
    exp_folder = exp.experimentation_path()

    loaded_exp = Experiment.load_breakpoint(os.path.join(exp_folder, 'breakpoint_0002'))
    loaded_exp.run_once(increase=True)


    clear_experiment_data(exp)
    clear_experiment_data(loaded_exp)


def test_03_sklearn_sgdclassfier():
    """Tests SGDClassifier"""

    n_features = 20
    n_classes = 3

    model_args = {'max_iter':1000, 'tol': 1e-3 ,
                   'n_features' : n_features, 'n_classes' : n_classes}

    training_args = {
        'epochs': 5,
        'loader_args': { 'batch_size': 1, },
    }

    tags =  ['#csv-dataset-classification']
    rounds = 2

    # search for corresponding datasets across nodes datasets
    exp = Experiment(tags=tags,
                     model_args=model_args,
                     training_plan_class=SGDClassifierTrainingPlan,
                     training_args=training_args,
                     round_limit=rounds,
                     aggregator=FedAverage(),
                     save_breakpoints=True)
    exp.run()
    clear_experiment_data(exp)


def test_04_sklearn_mnist_perceptron_with_declearn_optimizer():
    """Tests SGD classifier with Declearn optimizers"""


    model_args = {'n_features': 28*28,
                  'n_classes' : 10,
                  'eta0':1e-6,
                  'random_state':1234,
                  'alpha':0.1 }

    training_args = {
        'epochs': 3,
        'batch_maxnum': 20,
        'optimizer_args': {
            "lr" : 1e-3
        },
        'loader_args': { 'batch_size': 4, },
    }

    tags =  ['#MNIST', '#dataset']
    rounds = 4

    # select nodes participating in this experiment
    exp = Experiment(tags=tags,
                     model_args=model_args,
                     training_plan_class=SkLearnClassifierTrainingPlanDeclearn,
                     training_args=training_args,
                     round_limit=rounds,
                     aggregator=FedAverage(),
                     node_selection_strategy=None,
                     save_breakpoints=True
                     )

    exp.run()
    exp_folder = exp.experimentation_path()
    del exp

    loaded_exp = Experiment.load_breakpoint(os.path.join(exp_folder, 'breakpoint_0001'))

    # Run starting from a breakpoints
    loaded_exp.run_once(increase=True)
    clear_experiment_data(loaded_exp)



def test_05_sklearn_adni_regressor_with_declearn_optimizer():
    """Tests declearn optimizer with sgd regressor"""

    RANDOM_SEED = 1234
    model_args = {
        'max_iter':2000,
        'tol': 1e-5,
        'eta0':0.05,
        'n_features': 6,
        'random_state': RANDOM_SEED
    }

    training_args = {
        'epochs': 5,
        'loader_args': { 'batch_size': 32, },
        'test_ratio':.3,
        'test_metric': MetricTypes.MEAN_SQUARE_ERROR,
        'test_on_local_updates': True,
        'test_on_global_updates': True
    }

    tags =  ['#adni']
    rounds = 5

    # select nodes participating to this experiment
    exp = Experiment(tags=tags,
                     model_args=model_args,
                     training_plan_class=SGDRegressorTrainingPlanDeclearn,
                     training_args=training_args,
                     round_limit=rounds,
                     aggregator=FedAverage(),
                     node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)
