import os
import time
import pytest

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
    # node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    # node_2 = create_component(ComponentType.NODE, config_name="config_n2.ini")

    researcher = create_component(ComponentType.RESEARCHER, config_name="res.ini")
    


    time.sleep(1)

    # Starts the nodes
    #node_processes, _ = start_nodes([node_1, node_2])

    # Clear files and processes created for the tests
    def clear():
        #kill_subprocesses(node_processes)

        print("Clearing component data")
        #clear_node_data(node_1)
        #clear_node_data(node_2)
        clear_researcher_data(researcher)

    # Good to wait 3 second to give time to nodes start
    print("Sleep 5 seconds. Giving some time for nodes to start")
    time.sleep(5)

    request.addfinalizer(clear)
    node_1 = None
    return node_1, researcher

#############################################
### Start writing tests
### Nodes will stay up till end of the tests
#############################################

def test_experiment_run_101_getting_started_notebook(setup):
    # FIXME: in the notebook, it is specified to run the following command:
    # `./scripts/fedbiomed_run node dataset add --mnist ${MNIST_INSTALL_PATH}`
    # however, this is not done through testing
    
    node_1, researcher = setup
    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }
    print("Adding first dataset --------------------------------------------")
    

    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")
    node_processes, _ = start_nodes([node_1,])
    add_dataset_to_node(node_1, dataset)
    execute_script(os.path.join(environ['ROOT_DIR'], 'notebooks', '101_getting-started.ipynb'))

    kill_subprocesses(node_processes)
    clear_node_data(node_1)
