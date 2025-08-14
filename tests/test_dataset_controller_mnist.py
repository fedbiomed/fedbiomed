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
    instance.__len__.return_value = 10
    return instance


def test_mnist_controller_init_success(mocker, mock_torch_mnist, tmp_path):
    controller = MnistController(root=tmp_path, train=True, download=True)
    assert controller._controller_kwargs["name"] == "MNIST"
    assert controller._controller_kwargs["root"] == str(tmp_path)
    assert controller._controller_kwargs["train"] is True


def test_get_nontransformed_item(mocker, mock_torch_mnist, tmp_path):
    controller = MnistController(root=tmp_path)
    sample = controller._get_nontransformed_item(index=1)
    assert isinstance(sample["data"], Image.Image)
    assert isinstance(sample["target"], int)


def test_len_and_shape(mocker, mock_torch_mnist, tmp_path):
    controller = MnistController(tmp_path)
    assert len(controller) == 10
    shape = controller.shape()
    assert isinstance(shape, dict)
    assert all(item in shape for item in ["data", "target"])
    assert shape["data"] == {"size": (28, 28), "mode": "L"}
    assert shape["target"] == 1


def test_raises_on_dataset_failure(mocker, mock_torch_mnist, tmp_path):
    mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.MNIST",
        side_effect=RuntimeError("Fail"),
    )
    with pytest.raises(FedbiomedError):
        MnistController(root=tmp_path)
