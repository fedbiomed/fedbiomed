import numpy as np
import pytest
import torch

from fedbiomed.common.dataset import MedicalFolderDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class DummyNifti:
    def __init__(self, arr):
        self._arr = arr

    def get_fdata(self):
        return self._arr


class DummyController:
    def __init__(self, samples):
        self.samples = samples
        self.modalities = ["T1", "T2", "label"]

    def get_sample(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


@pytest.fixture
def sample_dict():
    arr1 = np.ones((3, 4, 5))
    arr2 = np.full((3, 4, 5), 2)
    arr_label = np.zeros((3, 4, 5))
    return [
        {
            "T1": DummyNifti(arr1),
            "T2": DummyNifti(arr2),
            "label": DummyNifti(arr_label),
            "demographics": {"age": 30, "sex": "M"},
        },
        {
            "T1": DummyNifti(arr1 * 2),
            "T2": DummyNifti(arr2 * 2),
            "label": DummyNifti(arr_label + 1),
            "demographics": {"age": 40, "sex": "F"},
        },
        {
            "T1": DummyNifti(arr1 * 3),
            "T2": DummyNifti(arr2 * 3),
            "label": DummyNifti(arr_label + 2),
            "demographics": {"age": 50, "sex": "O"},
        },
    ]


@pytest.mark.parametrize("data_modalities", ["T1", ["T1"], ["T1", "T2"]])
@pytest.mark.parametrize("target_modalities", [None, "label", ["label"]])
@pytest.mark.parametrize(
    "to_format", [DataReturnFormat.SKLEARN, DataReturnFormat.TORCH]
)
def test_init_and_getitem(
    monkeypatch, sample_dict, data_modalities, target_modalities, to_format
):
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )
    ds = MedicalFolderDataset(
        data_modalities=data_modalities,
        target_modalities=target_modalities,
        transform=None,
        target_transform=None,
    )
    ds._to_format = to_format
    ds._controller = DummyController(sample_dict)
    for idx in range(3):
        data, target = ds.__getitem__(idx)
        # Data checks
        expected_data_modalities = (
            [data_modalities] if isinstance(data_modalities, str) else data_modalities
        )
        for mod in expected_data_modalities:
            assert mod in data
            arr = data[mod]
            if to_format == DataReturnFormat.SKLEARN:
                assert isinstance(arr, np.ndarray)
                assert arr.shape == (3, 4, 5)
            else:
                assert isinstance(arr, torch.Tensor)
                assert arr.shape == torch.Size([3, 4, 5])
        # Target checks
        if target_modalities is None:
            assert target is None
        else:
            expected_target_modalities = (
                [target_modalities]
                if isinstance(target_modalities, str)
                else target_modalities
            )
            for mod in expected_target_modalities:
                assert mod in target
                arr = target[mod]
                if to_format == DataReturnFormat.SKLEARN:
                    assert isinstance(arr, np.ndarray)
                    assert arr.shape == (3, 4, 5)
                else:
                    assert isinstance(arr, torch.Tensor)
                    assert arr.shape == torch.Size([3, 4, 5])


@pytest.mark.parametrize(
    "data_modalities,target_modalities,transform,target_transform",
    [
        ("T1", "label", {"T1": lambda x: x}, {"label": lambda x: x}),
        (
            ["T1", "T2"],
            ["label"],
            {"T1": lambda x: x, "T2": lambda x: x},
            {"label": lambda x: x},
        ),
        (
            "T1",
            "label",
            lambda d: {k: v for k, v in d.items()},
            None,
        ),
        (
            ["T1", "T2"],
            ["label"],
            None,
            lambda d: {k: v for k, v in d.items()},
        ),
        (
            ["T1", "T2"],
            ["label"],
            None,
            None,
        ),
        (
            ["T1", "T2"],
            None,
            None,
            None,
        ),
    ],
)
def test_getitem_transform_types(
    monkeypatch,
    sample_dict,
    data_modalities,
    target_modalities,
    transform,
    target_transform,
):
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )
    ds = MedicalFolderDataset(
        data_modalities=data_modalities,
        target_modalities=target_modalities,
        transform=transform,
        target_transform=target_transform,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)
    for idx in range(3):
        data, target = ds.__getitem__(idx)
        assert set(data.keys()) == (
            {data_modalities}
            if isinstance(data_modalities, str)
            else set(data_modalities)
        )
        if target is not None:
            assert set(target.keys()) == (
                {target_modalities}
                if isinstance(target_modalities, str)
                else set(target_modalities)
            )


@pytest.mark.parametrize(
    "bad_transform",
    [
        {
            "T1": lambda x: (_ for _ in ()).throw(RuntimeError("fail")),
            "T2": lambda x: x,
        },
        lambda d: (_ for _ in ()).throw(RuntimeError("fail")),
    ],
)
def test_getitem_transform_error(monkeypatch, sample_dict, bad_transform):
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )
    ds = MedicalFolderDataset(
        data_modalities=["T1", "T2"],
        target_modalities=["label"],
        transform=bad_transform,
        target_transform={"label": lambda x: x},
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)
    with pytest.raises(FedbiomedError):
        ds.__getitem__(0)


def test_demographics_transform(monkeypatch, sample_dict):
    def demo_transform(demo):
        dict_transform = {"M": 0, "F": 1, "O": 2}
        demo["sex"] = dict_transform[demo["sex"]]
        return demo

    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )
    ds = MedicalFolderDataset(
        data_modalities=["T1", "demographics"],
        target_modalities=["label"],
        transform={"T1": lambda x: x, "demographics": demo_transform},
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)
    for idx in range(3):
        data, _ = ds.__getitem__(idx)
        assert "demographics" in data
        assert data["demographics"]["sex"] in {0, 1, 2}


@pytest.mark.parametrize(
    "modality_input,expected",
    [("T1", {"T1"}), (["T1"], {"T1"}), (["T1", "T2"], {"T1", "T2"})],
)
def test_normalize_modalities(modality_input, expected):
    assert MedicalFolderDataset._normalize_modalities(modality_input) == expected


@pytest.mark.parametrize(
    "transform,modalities",
    [
        (None, {"T1", "T2"}),
        (lambda x: x, {"T1"}),
        ({"T1": lambda x: x}, {"T1", "T2"}),
    ],
)
def test_validate_transform_types(transform, modalities):
    out = MedicalFolderDataset._validate_transform(transform, modalities)
    if isinstance(out, dict):
        assert all(callable(v) for v in out.values())
    else:
        assert callable(out)


def test_validate_transform_extra_modalities():
    # Should raise if transform has keys not in modalities
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform(
            {"T1": lambda x: x, "T3": lambda x: x}, {"T1", "T2"}
        )


def test_validate_transform_invalid_type():
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform("invalid_transform", {"T1"})


def test_validate_transform_target_transform_without_target_modalities():
    # Should raise if target_transform is provided but target_modalities is None
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset(
            data_modalities="T1",
            target_modalities=None,
            transform=None,
            target_transform=lambda x: x,
        )


def test_validate_format_and_transformations_data_not_dict():
    # Should raise if data is not a dict
    ds = MedicalFolderDataset(
        data_modalities="T1",
        target_modalities=None,
        transform=None,
        target_transform=None,
    )
    with pytest.raises(FedbiomedError):
        ds._validate_format_and_transformations(
            data=["T1"], transform={"T1": lambda x: x}
        )


def test_validate_format_and_transformations_transform_error(sample_dict):
    # Should raise if transform raises an error
    ds = MedicalFolderDataset(
        data_modalities="T1",
        target_modalities=None,
        transform=lambda d: (_ for _ in ()).throw(RuntimeError("fail")),
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    with pytest.raises(FedbiomedError):
        ds._validate_format_and_transformations(
            data={"T1": sample_dict[0]["T1"]}, transform=ds._transform
        )


def test_complete_initialization(monkeypatch, sample_dict):
    """Test the complete_initialization method"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )
    ds = MedicalFolderDataset(
        data_modalities=["T1", "T2"],
        target_modalities=["label"],
        transform=None,
        target_transform=None,
    )

    # Test successful initialization
    ds.complete_initialization(controller_kwargs={}, to_format=DataReturnFormat.SKLEARN)

    assert ds.to_format == DataReturnFormat.SKLEARN
    assert ds._controller is not None


def test_getitem_uninitialized_controller():
    """Test that __getitem__ raises error when controller is not initialized"""
    ds = MedicalFolderDataset(
        data_modalities="T1",
        target_modalities=None,
        transform=None,
        target_transform=None,
    )

    with pytest.raises(
        FedbiomedError, match="Dataset object has not completed initialization"
    ):
        ds.__getitem__(0)


@pytest.mark.parametrize(
    "data_modalities",
    ["", [], None],
)
def test_empty_data_modalities(data_modalities):
    """Test that empty data_modalities raises error"""
    with pytest.raises(FedbiomedValueError, match="`data_modalities` cannot be empty"):
        MedicalFolderDataset(
            data_modalities=data_modalities,
            target_modalities=None,
            transform=None,
            target_transform=None,
        )


def test_process_sample_data(monkeypatch, sample_dict):
    """Test the _process_sample_data method"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    ds = MedicalFolderDataset(
        data_modalities=["T1", "demographics"],
        target_modalities=None,
        transform={"T1": lambda x: x * 2, "demographics": lambda x: x},
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)

    # Test processing sample data
    sample = sample_dict[0]
    modalities = {"T1", "demographics"}

    result = ds._process_sample_data(sample, modalities, ds._transform, idx=0)

    assert "T1" in result
    assert "demographics" in result
    assert isinstance(result["T1"], np.ndarray)
    assert isinstance(result["demographics"], dict)
    # Check transform was applied (doubled)
    expected_t1 = np.ones((3, 4, 5)) * 2
    np.testing.assert_array_equal(result["T1"], expected_t1)


def test_process_sample_data_transform_error(monkeypatch, sample_dict):
    """Test _process_sample_data raises error when transform fails"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    ds = MedicalFolderDataset(
        data_modalities=["T1"],
        target_modalities=None,
        transform={"T1": lambda x: (_ for _ in ()).throw(RuntimeError("fail"))},
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)

    sample = sample_dict[0]
    modalities = {"T1"}

    with pytest.raises(
        FedbiomedError, match="Failed to apply.*transform.*to modality 'T1'"
    ):
        ds._process_sample_data(sample, modalities, ds._transform, idx=0)


def test_normalize_modalities_invalid_type():
    """Test _normalize_modalities with invalid input types"""
    with pytest.raises(FedbiomedError):
        MedicalFolderDataset._normalize_modalities(123)

    with pytest.raises(FedbiomedError):
        MedicalFolderDataset._normalize_modalities({"key": "value"})

    with pytest.raises(FedbiomedError):
        MedicalFolderDataset._normalize_modalities([1, 2, 3])


def test_validate_transform_non_callable_values():
    """Test _validate_transform with non-callable values in dict"""
    with pytest.raises(FedbiomedValueError, match="dict must map strings to callables"):
        MedicalFolderDataset._validate_transform({"T1": "not_callable"}, {"T1"})


def test_native_to_framework_conversions(sample_dict):
    """Test the _native_to_framework conversion functions"""
    dummy_nifti = sample_dict[0]["T1"]

    # Test SKLEARN conversion
    sklearn_func = MedicalFolderDataset._native_to_framework[DataReturnFormat.SKLEARN]
    result_sklearn = sklearn_func(dummy_nifti)
    expected = dummy_nifti.get_fdata()
    np.testing.assert_array_equal(result_sklearn, expected)

    # Test TORCH conversion
    torch_func = MedicalFolderDataset._native_to_framework[DataReturnFormat.TORCH]
    result_torch = torch_func(dummy_nifti)
    expected_torch = torch.from_numpy(dummy_nifti.get_fdata())
    assert torch.equal(result_torch, expected_torch)
    assert isinstance(result_torch, torch.Tensor)


def test_validate_format_and_transformations_target(monkeypatch, sample_dict):
    """Test _validate_format_and_transformations with target data"""
    ds = MedicalFolderDataset(
        data_modalities=["T1"],
        target_modalities=["label"],
        transform=None,
        target_transform={"label": lambda x: x},
    )
    ds._to_format = DataReturnFormat.SKLEARN

    # Mock the parent class methods
    monkeypatch.setattr(
        ds, "_validate_format_conversion", lambda x, extra_info: x.get_fdata()
    )
    monkeypatch.setattr(
        ds, "_validate_transformation", lambda data, transform, extra_info: None
    )

    # Test with target data
    target_data = {"label": sample_dict[0]["label"]}
    ds._validate_format_and_transformations(
        data=target_data, transform=ds._target_transform, is_target=True
    )


def test_whole_dict_transforms(monkeypatch, sample_dict):
    """Test whole-dict transforms for both data and target"""

    def data_transform(data_dict):
        # Transform that uses entire dict and returns a single array
        result = data_dict["T1"] + data_dict["T2"]  # Sum T1 and T2
        return result

    def target_transform(target_dict):
        # Transform that modifies target dict
        result = {}
        for k, v in target_dict.items():
            result[k] = v + 1  # Add 1 to all target data
        return result

    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    ds = MedicalFolderDataset(
        data_modalities=["T1", "T2"],
        target_modalities=["label"],
        transform=data_transform,
        target_transform=target_transform,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)

    data, target = ds.__getitem__(0)

    # Check that whole-dict transforms were applied
    assert isinstance(data, np.ndarray)
    assert "label" in target

    # Verify transforms were applied (doubled for data, +1 for target)
    expected_data = np.ones((3, 4, 5)) + np.full((3, 4, 5), 2)
    expected_label = np.zeros((3, 4, 5)) + 1  # Original + 1

    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(target["label"], expected_label)


def test_process_sample_data_with_target(monkeypatch, sample_dict):
    """Test _process_sample_data with target processing"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    ds = MedicalFolderDataset(
        data_modalities=["T1"],
        target_modalities=["label"],
        transform=None,
        target_transform={"label": lambda x: x * 10},
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)

    sample = sample_dict[0]
    target_modalities = {"label"}

    # Test target processing
    result = ds._process_sample_data(
        sample, target_modalities, ds._target_transform, idx=0, is_target=True
    )

    assert "label" in result
    assert isinstance(result["label"], np.ndarray)
    # Check transform was applied (multiplied by 10)
    expected = np.zeros((3, 4, 5)) * 10
    np.testing.assert_array_equal(result["label"], expected)


def test_process_sample_data_whole_dict_transform_error(monkeypatch, sample_dict):
    """Test _process_sample_data with whole-dict transform that fails"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    def failing_transform(data_dict):
        raise RuntimeError("Transform failed")

    ds = MedicalFolderDataset(
        data_modalities=["T1"],
        target_modalities=None,
        transform=failing_transform,
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)

    sample = sample_dict[0]
    modalities = {"T1"}

    with pytest.raises(FedbiomedError, match="Failed to apply.*transform.*to sample"):
        ds._process_sample_data(sample, modalities, ds._transform, idx=0)


def test_torch_format_conversion(monkeypatch, sample_dict):
    """Test format conversion for TORCH format"""
    monkeypatch.setattr(
        MedicalFolderDataset,
        "_init_controller",
        lambda self, controller_kwargs: setattr(
            self, "_controller", DummyController(sample_dict)
        ),
    )

    ds = MedicalFolderDataset(
        data_modalities=["T1"],
        target_modalities=["label"],
        transform=None,
        target_transform=None,
    )
    ds._to_format = DataReturnFormat.TORCH
    ds._controller = DummyController(sample_dict)

    data, target = ds.__getitem__(0)

    assert isinstance(data["T1"], torch.Tensor)
    assert isinstance(target["label"], torch.Tensor)
    assert data["T1"].shape == torch.Size([3, 4, 5])
    assert target["label"].shape == torch.Size([3, 4, 5])
