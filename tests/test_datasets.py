import pytest

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.researcher.datasets import FederatedDataset

data = {
    "node-1": [{"dataset_id": "dataset-id-1", "shape": [100, 100]}],
    "node-2": [{"dataset_id": "dataset-id-2", "shape": [120, 120], "test_ratio": 0.0}],
}


# before the tests
@pytest.fixture
def fds():
    return FederatedDataset(data)


def test_federated_dataset_01_create_error():
    """Testing creation with incorrect data"""
    # prepare
    data_list = [
        3,
        (2,),
        [],
    ]

    for data in data_list:
        # test + check
        with pytest.raises(FedbiomedError):
            FederatedDataset(data)


def test_federated_dataset_02_data(fds):
    """Testing property .data()"""

    updated_data = fds.data()
    # federated dataset should have added a new entry `test_ratio` in the FederatedDataset
    assert data == updated_data, "Can not get data properly from FederatedDataset"


def test_federated_dataset_04_node_ids(fds):
    """Testing node_ids getter/properties
    FIXME: When refactoring properties as getters
    """

    node_ids = fds.node_ids()
    assert node_ids == ["node-1", "node-2"], (
        "Can not get node ids of FederatedDataset properly"
    )


def test_federated_dataset_05_sample_sizes(fds):
    """Testing node_ids getter/properties
    FIXME: When refactoring properties as getters
    """
    # Nothing to do it is an empty method
    sizes = [val["shape"][0] for (_, val) in data.items()]
    sample_sizes = fds.sample_sizes()
    assert sizes == sample_sizes, (
        "Provided sample sizes and result of sample_sizes do not match"
    )


def test_federated_dataset_06_shapes(fds):
    """Testing shapes property of FederatedDataset"""

    node_1 = list(data.keys())[0]
    node_2 = list(data.keys())[1]

    size_1 = data[node_1]["shape"][0]
    size_2 = data[node_2]["shape"][0]

    shapes = fds.shapes()
    assert shapes[node_1] == size_1
    assert shapes[node_2] == size_2
