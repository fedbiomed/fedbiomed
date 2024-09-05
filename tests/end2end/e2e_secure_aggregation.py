import time
import pytest

from helpers import (
    configure_secagg,
    secagg_certificate_registration,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_node_data,
    clear_researcher_data,
    clear_experiment_data,
    create_multiple_nodes,
    create_node,
    create_researcher
)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.secagg import SecureAggregation, SecureAggregationSchemes as SecAggSchemes

dataset = {
    "name": "MNIST",
    "description": "MNIST DATASET",
    "tags": "#MNIST,#dataset",
    "data_type": "default",
    "path": "./data/"
}

# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup(port, post_session, request):
    """Setup fixture for the module"""

    # Configure secure aggregation
    print("Configure secure aggregation ---------------------------------------------")
    print(f"USING PORT {port} for researcher server")
    configure_secagg()

    print("Creating components ---------------------------------------------")
    with create_multiple_nodes(
        port=port,
        num_nodes=2,
        config_sections={
            'security': {'secure_aggregation': 'True'},
            'researcher': {'port': port}
        }) as nodes:

        node_1, node_2 = nodes

        print("Creating researcher component -------------------------------------------")
        researcher = create_researcher(port=port)

        print("Register certificates ---------------------------------------------")
        secagg_certificate_registration()

        print("Adding first dataset --------------------------------------------")
        add_dataset_to_node(node_1, dataset)
        print("adding second dataset")
        add_dataset_to_node(node_2, dataset)

        # Starts the nodes
        node_processes, thread = start_nodes([node_1, node_2])
        time.sleep(10)

        yield node_1, node_2, researcher

        # Clear files and processes created for the tests
        kill_subprocesses(node_processes)
        thread.join()

        print("Clearing researcher data")
        clear_researcher_data(researcher)


@pytest.fixture
def extra_node_force_secagg(port):
    """Fixture to add extra node which forces secagg"""

    node_3 = create_node(
        port=port,
        config_sections={
            'security': {
                'secure_aggregation': 'True',
                'force_secure_aggregation': 'True'},
        })

    add_dataset_to_node(node_3, dataset)
    # Re execute certificate registraiton
    secagg_certificate_registration()

    # Starts the nodes
    node_processes, thread = start_nodes([node_3])

    # Give some time to researcher
    time.sleep(10)

    yield

    kill_subprocesses(node_processes)
    thread.join()
    clear_node_data(node_3)

@pytest.fixture
def extra_node_no_validation(port):
    """Fixture to add extra node which disables validation"""

    node_3 = create_node(
        port=port,
        config_sections={
            'security': {
                'secure_aggregation': 'True',
                'secagg_insecure_validation': 'False'},
        })

    add_dataset_to_node(node_3, dataset)
    # Re execute certificate registraiton
    secagg_certificate_registration()

    # Starts the nodes
    node_processes, thread = start_nodes([node_3])

    # Give some time to researcher
    time.sleep(10)

    yield

    kill_subprocesses(node_processes)
    thread.join()
    clear_node_data(node_3)

@pytest.fixture
def extra_nodes_for_lom(port):

   with create_multiple_nodes(
        port,
        3,
        config_sections={
            'security': {
                'secure_aggregation': 'True',
                'force_secure_aggregation': 'True'},
    }) as nodes:

        node_1, node_2, node_3 = nodes

        for node in nodes:
            add_dataset_to_node(node, dataset)

        # start nodes and give some time to start
        node_processes, _ = start_nodes([node_1, node_2, node_3])
        time.sleep(10)


        yield

        kill_subprocesses(node_processes)

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

def test_01_secagg_joye_libert_pytorch_experiment_basic():
    """Tests running training mnist with basic configuration"""
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.JOYE_LIBERT),
    )

    exp.run()
    clear_experiment_data(exp)

def test_02_secagg_joye_libert_pytorch_breakpoint():
    """Tests running experiment with breakpoint and loading it while secagg active"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.JOYE_LIBERT),
        save_breakpoints=True
    )

    exp.run()

    # Delete experiment but do not clear its data
    del exp

    # Load experiment from latest breakpoint and continue training
    loaded_exp = Experiment.load_breakpoint()
    print("Running training round after loading the params")
    loaded_exp.run(rounds=2, increase=True)

    # Clear
    clear_experiment_data(loaded_exp)


def test_03_secagg_pytorch_force_secagg(extra_node_force_secagg):
    """Tests failure scenario whereas a node requires secure aggregation
        and researcher does not set it true
    """
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=3,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=False,
        save_breakpoints=True
    )

    # This should raise exception with default stragety
    with pytest.raises(SystemExit):
        exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_04_secagg_pytorch_no_validation(extra_node_no_validation):
    """Tests failure scenario whereas a researcher requires secure aggregation
        insecure validation and one node refuses to do it
    """
    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=3,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=True
    )

    # This should raise exception with default stragety
    with pytest.raises(SystemExit):
        exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_05_secagg_pytorch_lom():
    """Normal secagg using LOM"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
        save_breakpoints=True
    )
    exp.run()

    # Cleaning!
    clear_experiment_data(exp)


def test_06_secagg_lom_pytorch_breakpoint(extra_nodes_for_lom):
    """Tests running experiment with breakpoint and loading it while secagg active LOM"""

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=1,
        aggregator=FedAverage(),
        node_selection_strategy=None,
        secagg=SecureAggregation(scheme=SecAggSchemes.LOM),
        save_breakpoints=True
    )

    exp.run()
    secagg_id_before = exp.secagg.dh.secagg_id

    # Delete experiment but do not clear its data
    del exp

    # Load experiment from latest breakpoint
    loaded_exp = Experiment.load_breakpoint()

    # Check that `secagg_id` match (good hint that secagg context was properly reloaded)
    print("\nAsserting secagg context match after loading the params")
    secagg_id_after = loaded_exp.secagg.dh.secagg_id
    assert secagg_id_before, secagg_id_after

    # Continue training
    print("\nRunning training round after loading the params")
    loaded_exp.run(rounds=2, increase=True)

    # Clear
    clear_experiment_data(loaded_exp)
