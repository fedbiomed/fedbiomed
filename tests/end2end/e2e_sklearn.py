"""
End to End test module for testing  Sklearn training plans using:

- Different sklearn models
- Breakpoints
- Training and validation
- Declearn optimizer (on the node side)
- Secure aggregation

"""

import os
import time
import pytest

from helpers import (
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_experiment_data,
    clear_researcher_data,
    create_researcher,
    get_data_folder,
    create_multiple_nodes,
    configure_secagg,
    secagg_certificate_registration
)

from experiments.training_plans.sklearn import (
    PerceptronTraining,
    SGDRegressorTrainingPlan,
    SGDClassifierTrainingPlan,
    SkLearnClassifierTrainingPlanDeclearn,
    SGDRegressorTrainingPlanDeclearn,
    SGDRegressorTrainingPlanDeclearnScaffold,
    SkLearnClassifierTrainingPlanCustomTesting
)

from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import YogiModule as FedYogi, ScaffoldServerModule

from sklearn import datasets
import numpy as np

# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    with create_multiple_nodes(port, 3, {'security': {'secure_aggregation': 'True'}}) as nodes:

        node_1, node_2, node_3 = nodes  # pylint: disable=unbalanced-tuple-unpacking

        # Create researcher component
        researcher = create_researcher(port=port)
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
        datafolder = get_data_folder('MNIST-e2e-test')

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

        # Starts the nodes
        node_processes, thread = start_nodes([node_1, node_2, node_3])
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
    if all(os.path.isfile(p) for p in (p1, p2, p3)):
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



RANDOM_SEED = 1234
regressor_model_args = {
    'max_iter':2000,
    'tol': 1e-5,
    'eta0':0.05,
    'n_features': 6,
    'random_state': RANDOM_SEED
}

per_model_args = {
    'max_iter':1000,
    'tol': 1e-3 ,
    'n_features' : 20,
    'n_classes' : 2
}

per_training_args = {
    'epochs': 5,
    'loader_args': { 'batch_size': 1 }
}



def test_01_sklearn_perceptron():
    """Tests sklearn perceptron"""

    # search for corresponding datasets across nodes datasets
    exp = Experiment(
        tags=['#csv-dataset-classification'],
        model_args=per_model_args,
        training_plan_class=PerceptronTraining,
        training_args=per_training_args,
        round_limit=2,
        aggregator=FedAverage(),
        node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)

def test_02_sklearn_perceptron_custom_testing():
    """Tests sklearn perceptron using custom testing function in training plan"""

    per_training_args.update(
        {
        'test_ratio': .3,
        'test_on_local_updates': True,
        'test_on_global_updates': True
        }
    )

    exp = Experiment(
        tags=['#csv-dataset-classification'],
        model_args=per_model_args,
        training_plan_class=PerceptronTraining,
        training_args=per_training_args,
        round_limit=2,
        aggregator=FedAverage(),
        node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)


def test_03_sklean_sgdregressor():
    """Test SGDRegressor using Adni dataset"""


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
                     model_args=regressor_model_args,
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


def test_04_sklearn_sgdclassfier():
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

declearn_model_args = {
    'n_features': 28*28,
    'n_classes' : 10,
    'eta0':1e-6,
    'random_state':1234,
    'alpha':0.1 }

declearn_training_args = {
    'epochs': 3,
    'batch_maxnum': 20,
    'optimizer_args': {
        "lr" : 1e-3 },
    'loader_args': { 'batch_size': 4, },
}


def test_05_sklearn_mnist_perceptron_with_declearn_optimizer():
    """Tests SGD classifier with Declearn optimizers"""

    # select nodes participating in this experiment
    exp = Experiment(tags=["#MNIST", "#dataset"],
                     model_args=declearn_model_args,
                     training_plan_class=SkLearnClassifierTrainingPlanDeclearn,
                     training_args=declearn_training_args,
                     round_limit=4,
                     aggregator=FedAverage(),
                     node_selection_strategy=None,
                     save_breakpoints=True
                     )

    exp.run()
    exp_folder = exp.experimentation_path()
    del exp

    loaded_exp = Experiment.load_breakpoint(os.path.join(exp_folder, 'breakpoint_0001'))

    # Run starting from a breakpoint
    loaded_exp.run_once(increase=True)
    clear_experiment_data(loaded_exp)


def test_06_sklearn_mnist_perceptron_with_declearn_optimizer_on_researcher_side():
    """Test declearn optimizer on researcher side"""

    exp = Experiment(
        tags=["#MNIST", "#dataset"],
        model_args=declearn_model_args,
        training_plan_class=SkLearnClassifierTrainingPlanDeclearn,
        training_args=declearn_training_args,
        round_limit=4,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(lr=.8, modules=[FedYogi()]),
        node_selection_strategy=None,
        save_breakpoints=True
    )

    exp.run()
    clear_experiment_data(exp)


# Define paramters
regressor_training_args = {
    'epochs': 5,
    'loader_args': { 'batch_size': 32, },
    'test_ratio':.3,
    'test_metric': MetricTypes.MEAN_SQUARE_ERROR,
    'test_on_local_updates': True,
    'test_on_global_updates': True
}



def test_07_sklearn_adni_regressor_with_declearn_optimizer():
    """Tests declearn optimizer with sgd regressor"""

    tags =  ['#adni']
    rounds = 5

    # select nodes participating to this experiment
    exp = Experiment(tags=tags,
                     model_args=regressor_model_args,
                     training_plan_class=SGDRegressorTrainingPlanDeclearn,
                     training_args=regressor_training_args,
                     round_limit=rounds,
                     aggregator=FedAverage(),
                     node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)

def test_08_sklearn_adni_regressor_with_scaffold():
    """Tests sgd regressor training plan using declearn scaffold"""

    tags =  ['#adni']
    rounds = 5

    regressor_training_args.update({'optimizer_args': {'lr': 0.001}})
    # select nodes participating to this experiment
    exp = Experiment(
        tags=tags,
        model_args=regressor_model_args,
        training_plan_class=SGDRegressorTrainingPlanDeclearnScaffold,
        training_args=regressor_training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        agg_optimizer=Optimizer(lr=.8, modules=[ScaffoldServerModule()]),
        node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)



def test_09_seklearn_adni_regressor_with_secureaggregation():
    """Test SGDRegressor by activating secure aggregation"""

    # Configure secure aggregation setup
    configure_secagg()


    print("Register certificates ---------------------------------------------")
    secagg_certificate_registration()


    # select nodes participating to this experiment
    exp = Experiment(tags=['#adni'],
                     model_args=regressor_model_args,
                     training_plan_class=SGDRegressorTrainingPlanDeclearn,
                     training_args=regressor_training_args,
                     round_limit=5,
                     aggregator=FedAverage(),
                     secagg=True,
                     node_selection_strategy=None)

    exp.run()

    clear_experiment_data(exp)

