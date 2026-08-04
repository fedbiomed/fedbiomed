from unittest.mock import patch

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._mappings import get_controller
from fedbiomed.common.exceptions import FedbiomedError


class _FakeController:
    def __init__(self, root, param1=None):
        self.root = root
        self.param1 = param1


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_success(mock_registry):
    """Forwards the params accepted by the controller constructor"""
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (_FakeController, object)

    result = get_controller("csv", {"root": "/tmp/mock", "param1": "value1"})

    assert isinstance(result, _FakeController)
    assert result.root == "/tmp/mock"
    assert result.param1 == "value1"


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_drops_unknown_and_none(mock_registry):
    """Unknown keys and None values are not forwarded to the controller"""
    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (_FakeController, object)

    result = get_controller(
        "csv", {"root": "/tmp/mock", "param1": None, "unexpected": "x"}
    )

    assert result.root == "/tmp/mock"
    assert result.param1 is None  # default kept; None and unknown key dropped


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_unknown_type(mock_registry):
    mock_registry.__contains__.return_value = False
    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("unknown_type", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unknown 'data_type'" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_instantiation_error(mock_registry):
    """Generic exceptions during instantiation are wrapped in a FedbiomedError"""

    class _Boom:
        def __init__(self, root):
            raise Exception("Init error")

    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (_Boom, object)

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {"root": "/tmp"})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unhandled exception occurred" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_fedbiomed_error_propagation(mock_registry):
    """A FedbiomedError raised by the controller is propagated as is"""
    fb_error = FedbiomedError("Original error")

    class _Boom:
        def __init__(self, root):
            raise fb_error

    mock_registry.__contains__.return_value = True
    mock_registry.__getitem__.return_value = (_Boom, object)

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {"root": "/tmp"})
    assert excinfo.value is fb_error
