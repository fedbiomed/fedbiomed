import pytest
from PIL import Image

from fedbiomed.common.dataset_controller import MnistController
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_torch_mnist(mocker):
    mock_mnist = mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.MNIST"
    )
    instance = mock_mnist.return_value
    instance.__getitem__.return_value = (
        Image.new("L", (28, 28), color=128),
        0,
    )
    return instance


def test_init_with_valid_path(tmp_path, mock_torch_mnist):
    controller = MnistController(tmp_path)
    assert controller.root == tmp_path.resolve()


def test_init_with_valid_string_path(tmp_path, mock_torch_mnist):
    controller = MnistController(str(tmp_path))
    assert controller.root == tmp_path.resolve()


def test_init_with_nonexistent_path(mocker, mock_torch_mnist):
    mocker.patch(
        "fedbiomed.common.dataset_controller._controller.Path.exists",
        return_value=False,
    )
    with pytest.raises(FedbiomedError):
        MnistController("/nonexistent/path")


def test_shape(tmp_path, mock_torch_mnist):
    controller = MnistController(tmp_path)
    shape = controller.shape()
    assert isinstance(shape, dict)
    assert all(item in shape for item in ["data", "target"])
    print(shape)
    assert shape["data"] == {"size": (28, 28), "mode": "L"}
    assert shape["target"] == 1


def test_init_with_invalid_type(mock_torch_mnist):
    with pytest.raises(FedbiomedError):
        MnistController(1234)
