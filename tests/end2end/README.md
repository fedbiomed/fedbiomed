# End-to-end test

## Introduction: what are end-to-end tests?

In opposition to **unit-tests**, which are basically made for testing  small components or functionalities of the code, **End-to-end tests** are tests that test the whole functionality of the software, as if an end-user was using the software. It therfore includes secure aggregation facility configuration, loading datasets, loading certificates, ... .

Hence an **end-to-end** testing facility won't requiere fakes or Mocks, but the real components. For **end-to-end tests**, you can just make sure the usage of the software won't fail, you don't need to do assertions for each test.

## Material

End-to-end tests are run with [pytest](https://docs.pytest.org/) as test framework, as well as methods designed in `tests/end2end/helpers` folder.

## How to run tests

* setup the python environment

```
source ../scripts/fedbiomed_environment researcher
```
***
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


### writing en2end test

* `setup` method: is run at the begining of the tests. It contains instructions on how to set up Components. Methods for `helpers.py` can be used here for setting up Nodes and dataset

* `test_experiment_run_xxx` methods: contains the instructions of the tests

* `training plans` should be seperated from the tests (and defined in the folder `tests/end2end/experiments/training_plans`)

The basic structure of an **end-to-end test** file is the following:

```
import pytest
from helpers import (
    create_component,
    add_dataset_to_node,
    start_nodes,
    kill_subprocesses,
    clear_node_data,
    clear_experiment_data)

from experiments.training_plans.mnist_pytorch_training_plan import MyTrainingPlan
from fedbiomed.researcher.experiment import Experiment

@pytest.fixture(scope="module", autouse=True)
def setup(request):
    '''
    before each individual test
    '''
    node_1 = create_component(ComponentType.NODE, config_name="config_n1.ini")  # create one node

    mnist_dataset = {
        "name": "MNIST",
        "description": "MNIST DATASET",
        "tags": "#MNIST,#dataset",
        "data_type": "default",
        "path": "./data/"
    }
    add_dataset_to_node(node_1, mnist_dataset) 


def test_XXX_01_whatever(self):
    '''
    test the whatever feature
    '''
    exp = Experiment(
        ...)

    exp.run()

    clear_experiment_data(exp)  # remove folder created when running an Experiment

    # some code
    ...

```