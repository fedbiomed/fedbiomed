import pytest
from PIL import Image

from fedbiomed.common.dataset_controller import MnistController, download_mnist
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
    controller = MnistController(root=tmp_path, train=True)
    # The entry locates the dataset; it does not carry how it was obtained
    assert controller._controller_kwargs == {"root": str(tmp_path)}


@pytest.mark.parametrize(
    "supplied,expected",
    [({}, False), ({"download": False}, False), ({"download": True}, True)],
)
def test_mnist_controller_downloads_only_when_asked(
    mocker, mock_torch_mnist, tmp_path, supplied, expected
):
    """Without the flag, reading a deployed dataset must not reach the network.

    The database entry asserts the files are under `root`, so a missing file is
    an error to report rather than a reason to fetch during a training round.
    A caller that does want a fetch says so, and is obeyed.
    """
    mnist = mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.MNIST",
        return_value=mock_torch_mnist,
    )

    MnistController(root=tmp_path, **supplied)

    assert mnist.call_args.kwargs["download"] is expected


def test_get_sample(mocker, mock_torch_mnist, tmp_path):
    controller = MnistController(root=tmp_path)
    sample = controller.get_sample(index=1)
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


def test_download_mnist_fetches_into_root(mocker, tmp_path):
    """Deploying is the one step that may fetch the files."""
    mnist = mocker.patch("fedbiomed.common.dataset_controller._mnist_controller.MNIST")

    download_mnist(tmp_path)

    assert mnist.call_args.kwargs == {"root": tmp_path, "download": True}


def test_download_mnist_wraps_failures(mocker, tmp_path):
    mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.MNIST",
        side_effect=RuntimeError("no network"),
    )
    with pytest.raises(FedbiomedError, match="downloading MNIST"):
        download_mnist(tmp_path)
