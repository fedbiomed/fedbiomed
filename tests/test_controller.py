import pytest
import torch

from fedbiomed.common.dataset_controller import MnistController
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_mnist(mocker):
    mock_dataset = mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.datasets.MNIST"
    )
    instance = mock_dataset.return_value
    instance.data = torch.zeros((10, 28, 28))
    instance.targets = torch.arange(10)
    return instance


def test_init_with_valid_path(tmp_path, mock_mnist):
    controller = MnistController(tmp_path)
    assert controller.root == tmp_path.resolve()


def test_init_with_valid_string_path(tmp_path, mock_mnist):
    controller = MnistController(str(tmp_path))
    assert controller.root == tmp_path.resolve()


def test_init_with_nonexistent_path(mocker, mock_mnist):
    mocker.patch(
        "fedbiomed.common.dataset_controller._controller.Path.exists",
        return_value=False,
    )
    with pytest.raises(FedbiomedError):
        MnistController("/nonexistent/path")


def test_init_with_invalid_type(mock_mnist):
    with pytest.raises(FedbiomedError):
        MnistController(1234)
