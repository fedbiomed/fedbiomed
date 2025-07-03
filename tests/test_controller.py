import pytest
import torch

from fedbiomed.common.dataset_controller import MnistController
from fedbiomed.common.dataset_types import DataType
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_mnist_controller(mocker):
    mock_dataset = mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.datasets.MNIST"
    )
    instance = mock_dataset.return_value
    instance.data = torch.zeros((10, 28, 28))
    instance.targets = torch.arange(10)
    return instance


def test_init_with_valid_path(tmp_path, mock_mnist_controller):
    controller = MnistController(tmp_path)
    assert controller.root == tmp_path.resolve()


def test_init_with_valid_string_path(tmp_path, mock_mnist_controller):
    controller = MnistController(str(tmp_path))
    assert controller.root == tmp_path.resolve()


def test_init_with_nonexistent_path(mocker, mock_mnist_controller):
    mocker.patch(
        "fedbiomed.common.dataset_controller._controller.Path.exists",
        return_value=False,
    )
    with pytest.raises(FedbiomedError):
        MnistController("/nonexistent/path")


def test_shape(tmp_path, mock_mnist_controller):
    controller = MnistController(tmp_path)
    shape = controller.shape()
    assert shape[0] == 10
    assert isinstance(shape[1], dict)
    assert "data" in shape[1]
    assert shape[1]["data"] == (DataType.IMAGE, (28, 28))
    assert isinstance(shape[2], dict)
    assert "target" in shape[2]
    assert shape[2]["target"] == (DataType.TABULAR, (1,))


def test_init_with_invalid_type(mock_mnist_controller):
    with pytest.raises(FedbiomedError):
        MnistController(1234)
