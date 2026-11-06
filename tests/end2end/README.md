# End-to-end test

## Introduction: what are end-to-end tests?

In opposition to **unit tests**, which are basically made for testing small components or functionalities of the code, **End-to-end tests** are tests that test the whole functionality of the software, as if an end-user was using the software. It therefore may include secure aggregation facility configuration, loading datasets, loading certificates, and every other upcoming or existing functionalities.

Hence an **end-to-end** testing facility won't require fakes or Mocks, but the real components. For **end-to-end tests**, you can just make sure the usage of the software won't fail and that the final results are correct, you don't need to do assertions for each test.

## Material

End-to-end tests are run with [pytest](https://docs.pytest.org/) as test framework, as well as methods designed in `tests/end2end/helpers` folder.

## How to run tests

* use the python environment for [development](../docs/developer/development-environment.md)

* run all tests

```
cd tests
pytest -s -v end2end/e2e_*.py
```

* run a specific test file

```
cd tests
pytest -s -v end2end/e2e_xxxx.py
```

* run a specific test
(for instance `test_experiment_run_01` in `tests/end2end/e2e_mnist_pytorch.py` end-to-end test file)

```
cd tests
pytest -s -v end2end/e2e_mnist_pytorch.py::test_experiment_run_01
```
## How to write end-to-end tests

### Naming convention

Tests file should be located in the folder `tests/end2end/`. They should be named with the `e2e` prefix:
 for instance: `e2e_my_test.py`


### Writing end2end test

* `setup` method: is run at the beginning of the tests of a single test file. It requests the `federation` fixture and declares the nodes and datasets the module needs.

* `test_experiment_run_xxx` methods: contains the instructions of the tests

* `training plans` should be separated from the tests (and defined in the folder `tests/end2end/experiments/training_plans`)

The basic structure of an **end-to-end test** file is the following:

```python
import time

import pytest
from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan
from helpers import add_dataset_to_node, clear_experiment_data, get_data_folder

from fedbiomed.researcher.federated_workflows import Experiment


@pytest.fixture(scope="module", autouse=True)
def setup(federation):
    """Setup fixture for the module"""

    dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": get_data_folder("MNIST-e2e-test"),
    }

    with federation.nodes(2) as (node_1, node_2):
        add_dataset_to_node(node_1, dataset)
        add_dataset_to_node(node_2, dataset)

        federation.start((node_1, node_2))
        time.sleep(10)

        yield node_1, node_2, federation.researcher


def test_XXX_01_whatever():
    """test the whatever feature"""
    exp = Experiment(...)

    exp.run()

    clear_experiment_data(exp)
```

#### The federation

`federation` is a module-scoped fixture holding the researcher and the nodes of the test module. A process can run only one researcher, so a module builds one federation and adds nodes to it.

`federation.nodes(count, config_sections)` is a context manager. It creates the node components, and on leaving the block stops whatever was started inside it and removes those components. A function-scoped fixture can therefore add nodes for a single test:

```python
@pytest.fixture
def extra_node_force_secagg(federation):
    """Fixture to add extra node which forces secagg"""

    with federation.nodes(
        1, {"security": {"force_secure_aggregation": "True"}}
    ) as (node_3,):
        add_dataset_to_node(node_3, dataset)
        federation.start((node_3,))
        time.sleep(15)

        yield
```

`federation.start(nodes)` starts the node processes and keeps their supervisor, which is what reports a node that dies on its own. Nothing in a test kills processes or clears components by hand.

Datasets must be registered before the nodes start, because `dataset add` writes to a stopped node's database. `get_data_folder` returns a cached location, so a dataset is downloaded once and reused by every module and every run.

#### Post actions

Cleaning of the experiment after it is completed. **IT IS IMPORTANT** to call `clear_experiment_data` after `experiment.run` is completed. This method will make sure that the experiment data is deleted, and most importantly gRPC server is stopped before starting a new experiment. Please see following example code snippet that uses `clear_experiment_data`.

```python
def test_experiment_run_01():
    """Tests running training mnist with basic configuration"""
    model_args = {}
    tags = ['#MNIST', '#dataset']
    rounds = 1
    training_args = {
        'loader_args': { 'batch_size': 48, },
        'optimizer_args': {
            "lr" : 1e-3
        },
        'num_updates': 100,
        'dry_run': False,

    }

    exp = Experiment(
        tags=tags,
        model_args=model_args,
        training_plan_class=MyTrainingPlan,
        training_args=training_args,
        round_limit=rounds,
        aggregator=FedAverage(),
        node_selection_strategy=None,)

    exp.run()

    # Clean experiment data
    clear_experiment_data(exp)

```

### Dataset Description

The datasets are described as Python dict, and converted to JSON afterwards to be pass it to CLI command `dataset add --file`.

```
{
    "name": "Mednist data",
    "description": "Mednist",
    "tags": "mednist",
    "data_type": "images",
    "path": "$HOME/tmp/MedNIST"
}
```

You can use OS environment variables in this script via `os.environ`.


