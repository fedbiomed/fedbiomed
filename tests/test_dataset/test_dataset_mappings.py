from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset._mappings import (
    DATASET_CLASSES_PER_TYPE,
    ControllerParametersBase,
    MedicalFolderParameters,
    get_controller,
    validate_dataset_args,
)
from fedbiomed.common.exceptions import FedbiomedError


def test_controller_parameters_base_to_dict():
    """Test to_dict method removes None values"""
    params = ControllerParametersBase(root="/tmp/test")
    assert params.to_dict() == {"root": "/tmp/test"}

    @dataclass
    class TestParams(ControllerParametersBase):
        opt: str = None

    params_sub = TestParams(root="/tmp/test", opt=None)
    assert params_sub.to_dict() == {"root": "/tmp/test"}

    params_sub_val = TestParams(root="/tmp/test", opt="value")
    assert params_sub_val.to_dict() == {"root": "/tmp/test", "opt": "value"}


def test_controller_parameters_base_from_dict():
    """Test from_dict method filters extra keys"""
    data = {"root": "/tmp/test", "extra": "ignored"}
    params = ControllerParametersBase.from_dict(data)
    assert params.root == "/tmp/test"
    assert not hasattr(params, "extra")


def test_medical_folder_parameters():
    """Test MedicalFolderParameters specifics"""
    params = MedicalFolderParameters(root="/tmp/med")
    assert params.dlp is None
    assert params.tabular_file is None

    # Test with DLP
    dlp_mock = MagicMock(spec=DataLoadingPlan)
    params = MedicalFolderParameters(root="/tmp/med", dlp=dlp_mock)
    assert params.dlp == dlp_mock

    # Test conversion
    dct = params.to_dict()
    assert "dlp" in dct
    assert dct["dlp"] is not None

    # asdict deepcopies fields, so we check type rather than identity for the mock
    assert isinstance(dct["dlp"], MagicMock)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_success(mock_registry):
    """Test successful controller creation"""
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()
    mock_params_instance = MagicMock()
    mock_params_instance.to_dict.return_value = {"param1": "value1"}
    mock_params_cls.from_dict.return_value = mock_params_instance

    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.side_effect = lambda k: True

    result = get_controller("csv", {"param1": "value1"})

    mock_params_cls.from_dict.assert_called_once_with({"param1": "value1"})
    mock_controller_cls.assert_called_once_with(param1="value1")
    assert result == mock_controller_cls.return_value


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_unknown_type(mock_registry):
    """Test handling of unknown dataset type"""
    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("unknown_type", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unknown 'data_type'" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_param_parse_error(mock_registry):
    """Test error during parameter parsing"""
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.side_effect = Exception("Parse error")
    mock_registry.__getitem__.return_value = (MagicMock(), mock_params_cls, MagicMock())
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Failed to parse dataset_parameters" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_instantiation_error(mock_registry):
    """Test error during controller instantiation"""
    mock_controller_cls = MagicMock(side_effect=Exception("Init error"))
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.return_value.to_dict.return_value = {}
    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unhandled exception occurred" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_fedbiomed_error_propagation(mock_registry):
    """Test that FedbiomedError is propagated as is"""
    mock_controller_cls = MagicMock(side_effect=FedbiomedError("Original error"))
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.return_value.to_dict.return_value = {}
    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert "Original error" in str(excinfo.value)


@pytest.fixture
def mock_datasets():
    """Fixture to mock DATASET_CLASSES_PER_TYPE with controlled classes"""

    class MockDatasetValid:
        def __init__(self, req_arg, opt_arg=None):
            pass

    class MockDatasetWithCwd:
        def __init__(self, data):
            pass

    with patch.dict(
        DATASET_CLASSES_PER_TYPE,
        {
            DatasetTypes.TABULAR: MockDatasetValid,
            DatasetTypes.IMAGES: MockDatasetWithCwd,
        },
        clear=True,
    ):
        yield


def test_validate_args_success(mock_datasets):
    """Test valid arguments pass validation"""
    validate_dataset_args("csv", {"req_arg": 1})
    validate_dataset_args("csv", {"req_arg": 1, "opt_arg": 2})


def test_validate_args_unsupported_type():
    """Test validation fails for unsupported type"""
    with pytest.raises(FedbiomedError) as exc:
        validate_dataset_args("unknown", {})
    assert "Unsupported dataset type" in str(exc.value)


def test_validate_args_missing_required(mock_datasets):
    """Test validation fails when required arg is missing"""
    with pytest.raises(FedbiomedError) as exc:
        validate_dataset_args("csv", {"opt_arg": 1})
    assert "Missing required dataset_args" in str(exc.value)
    assert "req_arg" in str(exc.value)


def test_validate_args_invalid_key(mock_datasets):
    """Test validation fails when extra arg provided"""
    with pytest.raises(FedbiomedError) as exc:
        validate_dataset_args("csv", {"req_arg": 1, "extra_arg": 3})
    assert "Invalid dataset_args" in str(exc.value)
    assert "extra_arg" in str(exc.value)


def test_validate_args_ignores_self_class():
    """Test that 'self' argument is ignored during validation"""

    class MockSelf:
        def __init__(self, a):
            pass

    with patch.dict(
        DATASET_CLASSES_PER_TYPE, {DatasetTypes.TABULAR: MockSelf}, clear=True
    ):
        validate_dataset_args("csv", {"a": 1})


def test_validate_args_disallows_kwargs():
    """Test that validation fails (via invalid keys) even if class accepts kwargs,
    because strict validation ignores kwargs signature."""

    class MockKwargs:
        def __init__(self, a, **kwargs):
            pass

    with patch.dict(
        DATASET_CLASSES_PER_TYPE, {DatasetTypes.TABULAR: MockKwargs}, clear=True
    ):
        validate_dataset_args("csv", {"a": 1})

        with pytest.raises(FedbiomedError) as exc:
            validate_dataset_args("csv", {"a": 1, "b": 2})
        assert "Invalid dataset_args" in str(exc.value)
        assert "b" in str(exc.value)


def test_validate_args_real_datasets():
    """Test validation with actual dataset classes defined in the mappings."""
    # Test TabularDataset
    args_tabular = {"input_columns": [1, 2]}
    validate_dataset_args(DatasetTypes.TABULAR.value, args_tabular)

    # Test ImageFolderDataset
    args_images = {"transform": None}
    validate_dataset_args(DatasetTypes.IMAGES.value, args_images)

    # Test MedicalFolderDataset
    args_medical = {"data_modalities": ["T1"], "target_modalities": None}
    validate_dataset_args(DatasetTypes.MEDICAL_FOLDER.value, args_medical)

    # Test MedNistDataset
    args_mednist = {}
    validate_dataset_args(DatasetTypes.MEDNIST.value, args_mednist)

    # Test MnistDataset
    args_mnist = {}
    validate_dataset_args(DatasetTypes.DEFAULT.value, args_mnist)

    # Test CustomDataset
    args_custom = {}
    validate_dataset_args(DatasetTypes.CUSTOM.value, args_custom)
