import numpy as np
import pytest
import torch

from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import NipoppyDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from testsupport.fake_nipoppy_reader import DERIVATIVE, DERIVATIVE_COLUMN

# `dataset_parameters` stored in the node database at registration
STORED_PARAMETERS = {
    "available_phenotypes": ["age", "sex", "diagnosis"],
    "installed_pipelines": {
        "bidsification": {},
        "processing": {"freesurfer": ["7.3.2"]},
        "extraction": {"fs_stats": ["0.2.1"]},
    },
}


def make_dataset(**kwargs):
    defaults = dict(
        phenotypes=["age", "sex"],
        derivatives=None,
        target=["diagnosis"],
        session_filters="baseline",
    )
    defaults.update(kwargs)
    return NipoppyDataset(**defaults)


def node_load(dataset, root, to_format=DataReturnFormat.SKLEARN, **extra):
    """Loads the dataset with the kwargs the node passes (see `Round` / FA job)."""
    dataset.load(to_format=to_format, root=str(root), **STORED_PARAMETERS, **extra)


# ---------- load ----------


def test_load_with_node_kwargs_sklearn(tmp_path, fake_nipoppy_reader):
    ds = make_dataset()
    node_load(ds, tmp_path)

    # sub-03 dropped (missing age)
    assert len(ds) == 2
    data, target = ds[0]
    assert isinstance(data, np.ndarray) and data.dtype == np.float64
    assert isinstance(target, np.ndarray) and target.dtype == np.float64
    np.testing.assert_array_equal(data, [70.0, 0.0])
    np.testing.assert_array_equal(target, [1.0])


def test_load_with_node_kwargs_torch(tmp_path, fake_nipoppy_reader):
    ds = make_dataset()
    node_load(ds, tmp_path, to_format=DataReturnFormat.TORCH)

    data, target = ds[1]
    assert isinstance(data, torch.Tensor)
    assert data.dtype == torch.get_default_dtype()
    assert torch.equal(data, torch.tensor([65.0, 1.0]))
    assert torch.equal(target, torch.tensor([0.0]))


def test_load_forwards_parameters_to_controller(tmp_path, mocker, fake_nipoppy_reader):
    whole_df = mocker.Mock(side_effect=lambda df: df)
    ds = make_dataset(
        drop_na=True, drop_na_kwargs={"how": "any"}, whole_df_level_transform=whole_df
    )
    init_controller = mocker.spy(ds, "_init_controller")

    node_load(ds, tmp_path)

    # Metadata is kept by the dataset, not forwarded to the controller
    init_controller.assert_called_once_with(
        root=str(tmp_path),
        phenotypes=["age", "sex", "diagnosis"],
        derivatives=None,
        session_filters="baseline",
        drop_na=True,
        drop_na_kwargs={"how": "any"},
        whole_df_transform=whole_df,
        dlp=None,
    )
    whole_df.assert_called_once()


def test_load_reads_data_once(tmp_path, fake_nipoppy_reader):
    node_load(make_dataset(), tmp_path)

    assert len(fake_nipoppy_reader) == 1
    assert len(fake_nipoppy_reader[0].read_calls) == 1


def test_load_multiple_sessions_per_participant_raises(tmp_path, fake_nipoppy_reader):
    # sub-01 has a baseline and a followup session
    ds = make_dataset(session_filters=None)

    with pytest.raises(FedbiomedError, match="more than one session"):
        node_load(ds, tmp_path)


def test_load_with_dlp_raises(tmp_path, fake_nipoppy_reader):
    with pytest.raises(FedbiomedError, match="not supported"):
        node_load(make_dataset(), tmp_path, dlp=DataLoadingPlan())


def test_load_reads_target_once(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(phenotypes=["age", "diagnosis"], target="diagnosis")
    node_load(ds, tmp_path)

    assert fake_nipoppy_reader[0].read_calls[-1]["phenotypes"] == ["age", "diagnosis"]
    data, target = ds[0]
    np.testing.assert_array_equal(data, [70.0])
    np.testing.assert_array_equal(target, [1.0])


def test_load_with_derivatives(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(derivatives=[DERIVATIVE])
    node_load(ds, tmp_path)

    assert fake_nipoppy_reader[0].read_calls[-1] == {
        "phenotypes": ["age", "sex", "diagnosis"],
        "derivatives": [DERIVATIVE],
    }
    data, _ = ds[0]
    np.testing.assert_array_equal(data, [70.0, 0.0, 2.5])


def test_non_numeric_sample_raises_on_getitem(tmp_path, fake_nipoppy_reader):
    def to_string(df):
        df = df.copy()
        df["sex"] = df["sex"].map({0.0: "M", 1.0: "F"})
        return df

    ds = make_dataset(whole_df_level_transform=to_string)
    # Not detected by `load`: dtypes are only checked when applying default types
    node_load(ds, tmp_path)

    with pytest.raises(FedbiomedError) as exc_info:
        ds[0]
    assert "numeric" in str(exc_info.value.__cause__)


def test_deprecated_complete_initialization(tmp_path, fake_nipoppy_reader):
    ds = make_dataset()
    ds.complete_initialization(
        controller_kwargs={"root": str(tmp_path), **STORED_PARAMETERS},
        to_format=DataReturnFormat.SKLEARN,
    )

    assert len(ds) == 2


# ---------- target / phenotypes ----------


def test_no_target(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(target=None)
    node_load(ds, tmp_path)

    data, target = ds[0]
    np.testing.assert_array_equal(data, [70.0, 0.0])
    assert target is None


def test_phenotypes_none_uses_available_phenotypes(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(phenotypes=None)
    node_load(ds, tmp_path)

    assert fake_nipoppy_reader[0].read_calls[-1]["phenotypes"] == [
        "age",
        "sex",
        "diagnosis",
    ]
    data, target = ds[0]
    np.testing.assert_array_equal(data, [70.0, 0.0])
    np.testing.assert_array_equal(target, [1.0])


def test_phenotypes_none_without_available_phenotypes_raises(
    tmp_path, fake_nipoppy_reader
):
    ds = make_dataset(phenotypes=None)

    with pytest.raises(FedbiomedError, match="Specify `phenotypes`"):
        ds.load(root=str(tmp_path), to_format=DataReturnFormat.SKLEARN)


# ---------- metadata ----------


def test_get_metadata_before_load_is_empty():
    assert make_dataset().get_metadata() == {}


def test_get_metadata_returns_passed_values(tmp_path, fake_nipoppy_reader):
    ds = make_dataset()
    node_load(ds, tmp_path)

    assert ds.get_metadata() == STORED_PARAMETERS


def test_get_metadata_none_when_not_passed(tmp_path, fake_nipoppy_reader):
    ds = make_dataset()
    ds.load(root=str(tmp_path), to_format=DataReturnFormat.SKLEARN)

    assert ds.get_metadata() == {
        "available_phenotypes": None,
        "installed_pipelines": None,
    }


# ---------- derivatives initializer ----------


def _pick_freesurfer(dataset):
    pipelines = dataset.get_metadata()["installed_pipelines"]
    version = pipelines["processing"]["freesurfer"][0]
    dataset.derivatives = [("freesurfer", version, DERIVATIVE[2])]


@pytest.mark.parametrize("via_constructor", [True, False])
def test_derivatives_initializer_sets_derivatives_before_reading(
    tmp_path, fake_nipoppy_reader, via_constructor
):
    if via_constructor:
        ds = make_dataset(derivatives_initializer=_pick_freesurfer)
    else:
        ds = make_dataset()
        ds.set_derivatives_initializer(_pick_freesurfer)
    node_load(ds, tmp_path)

    assert ds.derivatives == [DERIVATIVE]
    assert fake_nipoppy_reader[0].read_calls == [
        {"phenotypes": ["age", "sex", "diagnosis"], "derivatives": [DERIVATIVE]}
    ]
    data, _ = ds[0]
    np.testing.assert_array_equal(data, [70.0, 0.0, 2.5])


def test_derivatives_initializer_error_is_wrapped(tmp_path, fake_nipoppy_reader):
    def failing(dataset):
        raise KeyError("boom")

    ds = make_dataset(derivatives_initializer=failing)

    with pytest.raises(FedbiomedError, match="derivatives initializer"):
        node_load(ds, tmp_path)
    assert fake_nipoppy_reader == []  # the controller was never created


def test_derivatives_initializer_not_callable_raises():
    with pytest.raises(FedbiomedValueError):
        make_dataset(derivatives_initializer="not callable")
    with pytest.raises(FedbiomedValueError):
        make_dataset().set_derivatives_initializer(42)


# ---------- transforms ----------


def test_sample_level_transforms_applied(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(
        sample_level_transform=lambda x: x * 2,
        sample_level_target_transform=lambda y: y + 10,
    )
    node_load(ds, tmp_path)

    data, target = ds[0]
    np.testing.assert_array_equal(data, [140.0, 0.0])
    np.testing.assert_array_equal(target, [11.0])


def test_sample_level_transform_wrong_output_raises_at_load(
    tmp_path, fake_nipoppy_reader
):
    ds = make_dataset(sample_level_transform=lambda x: x.tolist())

    with pytest.raises(FedbiomedError, match="transform"):
        node_load(ds, tmp_path)


def test_transform_not_callable_raises():
    with pytest.raises(FedbiomedValueError):
        make_dataset(sample_level_transform="not callable")
    with pytest.raises(FedbiomedValueError):
        make_dataset(sample_level_target_transform=1)


def test_drop_na_kwargs_ignored_warning(mocker):
    warning = mocker.patch("fedbiomed.common.dataset._nipoppy_dataset.logger.warning")

    make_dataset(drop_na=False, drop_na_kwargs={"how": "all"})

    warning.assert_called_once()


# ---------- getitem / iteration ----------


def test_getitem_before_load_raises():
    with pytest.raises(FedbiomedError, match="Call `load` first"):
        make_dataset()[0]


def test_len_matches_iteration(tmp_path, fake_nipoppy_reader):
    ds = make_dataset(phenotypes=["sex"], derivatives=[DERIVATIVE])
    node_load(ds, tmp_path)

    # sub-03 kept: its missing `age` is not selected
    items = list(ds)
    assert len(items) == len(ds) == 3
