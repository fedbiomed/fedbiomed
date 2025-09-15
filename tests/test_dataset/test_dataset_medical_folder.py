import numpy as np
import pytest
import torch

from fedbiomed.common.dataset._medical_folder_dataset import MedicalFolderDataset
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
        demographics_transform=None,
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
    "transform,target_transform",
    [
        ({"T1": lambda x: x, "T2": lambda x: x}, {"label": lambda x: x}),
        (
            lambda d: {k: v for k, v in d.items()},
            lambda d: {k: v for k, v in d.items()},
        ),
    ],
)
@pytest.mark.parametrize(
    "data_modalities,target_modalities",
    [
        ("T1", "label"),
        (["T1", "T2"], ["label"]),
    ],
)
def test_getitem_transform_types(
    monkeypatch,
    sample_dict,
    transform,
    target_transform,
    data_modalities,
    target_modalities,
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
        demographics_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    ds._controller = DummyController(sample_dict)
    for idx in range(3):
        data, target = ds.__getitem__(idx)
        assert (
            set(data.keys()) == {data_modalities}
            if isinstance(data_modalities, str)
            else set(data_modalities)
        )
        assert (
            set(target.keys()) == {target_modalities}
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
        demographics_transform=None,
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
        transform=None,
        target_transform=None,
        demographics_transform=demo_transform,
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
        (None, {"T1"}),
        (lambda x: x, {"T1"}),
        ({"T1": lambda x: x}, {"T1"}),
    ],
)
def test_validate_transform_types(transform, modalities):
    out = MedicalFolderDataset._validate_transform(transform, modalities)
    if isinstance(out, dict):
        assert all(callable(v) for v in out.values())
    else:
        assert callable(out)


def test_validate_transform_missing_modalities():
    with pytest.raises(FedbiomedError):
        MedicalFolderDataset._validate_transform({"T1": (lambda x: x)}, {"T1", "T2"})


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
            demographics_transform=None,
        )


def test_validate_transform_whole_dict_and_demographics_transform():
    # Should raise if both whole-dict transform and demographics_transform are provided
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform(
            transform=lambda d: d,
            modalities={"T1", "demographics"},
            demographics_transform=lambda x: x,
        )


def test_validate_transform_dict_and_demographics_transform_redundancy():
    # Should raise if both transform['demographics'] and demographics_transform are provided
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform(
            transform={"T1": lambda x: x, "demographics": lambda x: x},
            modalities={"T1", "demographics"},
            demographics_transform=lambda x: x,
        )


def test_validate_transform_demographics_transform_not_callable():
    # Should raise if demographics_transform is not callable
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform(
            transform={"T1": lambda x: x},
            modalities={"T1", "demographics"},
            demographics_transform="not_callable",
        )


def test_validate_transform_demographics_transform_missing_demographics():
    # Should raise if demographics_transform is provided but 'demographics' not in modalities
    with pytest.raises(FedbiomedValueError):
        MedicalFolderDataset._validate_transform(
            transform={"T1": lambda x: x},
            modalities={"T1"},
            demographics_transform=lambda x: x,
        )


def test_validate_format_and_transformations_data_not_dict():
    # Should raise if data is not a dict
    ds = MedicalFolderDataset(
        data_modalities="T1",
        target_modalities=None,
        transform=None,
        target_transform=None,
        demographics_transform=None,
    )
    with pytest.raises(FedbiomedError):
        ds._validate_format_and_transformations(
            data=["not", "a", "dict"], transform={"T1": lambda x: x}
        )


def test_validate_format_and_transformations_transform_error(monkeypatch, sample_dict):
    # Should raise if transform raises an error
    ds = MedicalFolderDataset(
        data_modalities="T1",
        target_modalities=None,
        transform=lambda d: (_ for _ in ()).throw(RuntimeError("fail")),
        target_transform=None,
        demographics_transform=None,
    )
    ds._to_format = DataReturnFormat.SKLEARN
    with pytest.raises(FedbiomedError):
        ds._validate_format_and_transformations(
            data={"T1": sample_dict[0]["T1"]}, transform=ds._transform
        )
