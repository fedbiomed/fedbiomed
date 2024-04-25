import time
import pytest

from helpers import (
    configure_secagg,
    secagg_certificate_registration,
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_node_data,
    clear_researcher_data,
    clear_experiment_data)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.common.constants import ComponentType
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold


dataset = {
    "name": "MNIST",
    "description": "MNIST DATASET",
    "tags": "#MNIST,#dataset",
    "data_type": "default",
    "path": "./data/"
}

# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(request):
    """Setup fixture for the module"""

    # Configure secure aggregation
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
    add_dataset_to_node(node_1, dataset)
    print("adding second dataset")
    add_dataset_to_node(node_2, dataset)

    time.sleep(1)

    # Starts the nodes
    node_processes, _ = start_nodes([node_1, node_2])

    # Clear files and processes created for the tests
    def clear():
        kill_subprocesses(node_processes)

        print("Cleareaniing component data")
        clear_node_data(node_1)
        clear_node_data(node_2)

        clear_researcher_data(researcher)

    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(5)

    request.addfinalizer(clear)


@pytest.fixture
def extra_node():
    """Fixture to add extra node"""

    node_3 = create_component(
        ComponentType.NODE,
        config_name="config_n3.ini",
        config_sections={
            'security': {
                'secure_aggregation': 'True',
                'force_secure_aggregation': 'True'}
        })

    add_dataset_to_node(node_3, dataset)
    # Re execute certificate registraiton
    secagg_certificate_registration()

    # Starts the nodes
    node_processes, _ = start_nodes([node_3])

    # Give some time to researcher
    time.sleep(10)

    yield

    kill_subprocesses(node_processes)
    clear_node_data(node_3)


#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

model_args = {}
tags = ['#MNIST', '#dataset']
rounds = 2
training_args = {
    'loader_args': { 'batch_size': 48, },
    'optimizer_args': {
        "lr" : 1e-3
    },
    'num_updates': 100,
    'dry_run': False,
}

def test_experiment_run_01():
    """Tests running training mnist with basic configuration"""
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=True,
    )

    exp.run()
    clear_experiment_data(exp)

def test_experiment_02_secagg_breakpoint():
    """Tests running experiment with breakpoint and loading it while secagg active"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=True,
        save_breakpoints=True
    )

    exp.run()

    # Delete experiment but do not clear its data
    del exp

    # Load experiment from latest breakpoint and continue training
    loaded_exp = Experiment.load_breakpoint()
    print("Running training round after loading the params")
    loaded_exp.run(rounds=2, increase=True)
    clear_experiment_data(loaded_exp)


def test_experiment_03_node_force_secagg(extra_node):
    """Tests failure scnarios whereas a node requires secure aggregation
        and researcher does not set it true
    """
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=False,
        save_breakpoints=True
    )

    # This should raise exception with default stragety
    # with pytest.raises(SystemExit):
    exp.run()

    # Cleaning!
    clear_experiment_data(exp)




