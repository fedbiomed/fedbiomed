import time
import pytest

from helpers import (
    clear_component_data,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_experiment_data,
    training_plan_operation,
    create_node,
    create_researcher,
    get_data_folder
)

from experiments.training_plans.mnist_model_approval import TrainingPlanApprovalTP

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage


# Set up nodes and start
@pytest.fixture(scope="module", autouse=True)
def setup_components(port, post_session, request):
    """Setup fixture for the module"""
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder('MNIST-e2e-test')

    }

    print(f"USING PORT {port} for researcher erver")
    print("Creating§ components ---------------------------------------------")
    node_1 = create_node(
        port=port,
        config_sections={
            'security': {'training_plan_approval': 'True'}
        })

    node_2 = create_node(
        port=port,
        config_sections={
            'security': {'training_plan_approval': 'True'}
        })


    print("Creating researcher component -----------------------------------------")
    researcher = create_researcher(port=port)

    print("Adding first dataset --------------------------------------------")
    add_dataset_to_node(node_1, dataset)
    print("adding second dataset")
    add_dataset_to_node(node_2, dataset)


    # Starts the nodes
    node_processes, thread = start_nodes([node_1, node_2])
    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(10)

    # Clear files and processes created for the tests
    def clear():
        kill_subprocesses(node_processes)
        thread.join()

        print("Clearing component data")
        clear_component_data(node_1)
        clear_component_data(node_2)

        clear_component_data(researcher)

    request.addfinalizer(clear)


    return node_1, node_2, researcher

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

model_args = {}
tags = ['#MNIST', '#dataset']
rounds =  1
training_args = {
    'loader_args': { 'batch_size': 48, },
    'optimizer_args': {
        "lr" : 1e-3
    },
    'num_updates': 100,
    'dry_run': False,
}


def test_01_training_plan_approval_failure_success_cases(setup_components):
    """Tests running training mnist with basic configuration"""


    print("Running test_experiment_run_01")
    node_1, node_2, researcher = setup_components

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=TrainingPlanApprovalTP,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,
    )

    # Training plan is not approved exp.run should fail
    with pytest.raises(SystemExit):
        exp.run()

    # Check status
    status = exp.check_training_plan_status()

    node_1_id = node_1.get('default', 'id')
    node_2_id = node_2.get('default', 'id')

    assert node_1_id in status
    assert node_2_id in status

    assert status[node_1_id].status == 'Not Registered'
    assert status[node_2_id].status == 'Not Registered'


    print('Sending approval request')
    reply = exp.training_plan_approve(description="Test training plan")

    assert node_1_id in reply
    assert node_2_id in reply

    tp_id_1 = reply[node_1_id]['training_plan_id']
    tp_id_2 = reply[node_2_id]['training_plan_id']

    assert tp_id_1 is not None
    assert tp_id_2 is not None

    # Recheck status results it should be pending
    # Check status
    status = exp.check_training_plan_status()
    assert status[node_1_id].status == 'Pending'
    assert status[node_2_id].status == 'Pending'



    # Approve training plans --------------------------------
    training_plan_operation(node_1, 'approve', tp_id_1)
    training_plan_operation(node_2, 'approve', tp_id_2)

    # Recheck status results it should be pending
    # Check status
    status = exp.check_training_plan_status()
    assert status[node_1_id].status == 'Approved'
    assert status[node_2_id].status == 'Approved'

    # Should be able to run training
    exp.run()


    # Reject training plans
    training_plan_operation(node_1, 'reject', tp_id_1)
    training_plan_operation(node_2, 'reject', tp_id_2)

    status = exp.check_training_plan_status()
    assert status[node_1_id].status == 'Rejected'
    assert status[node_2_id].status == 'Rejected'

    # Should not be able to run experiment with rejected training plan
    with pytest.raises(SystemExit):
        exp.run(rounds=2, increase=True)


    # Important always clear experiment  data
    clear_experiment_data(exp)

