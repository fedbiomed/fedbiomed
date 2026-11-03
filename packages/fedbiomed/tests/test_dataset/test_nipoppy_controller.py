import pytest

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import NipoppyDataset
from fedbiomed.common.dataset._mappings import REGISTRY_CONTROLLERS, get_controller
from fedbiomed.common.dataset_controller import NipoppyController
from fedbiomed.common.exceptions import FedbiomedError

from testsupport.fake_nipoppy_reader import DERIVATIVE, DERIVATIVE_COLUMN, FakeNipoppyReader

# ---------- construction ----------


def test_init_controller_kwargs_read_from_disk(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path)

    assert controller._controller_kwargs == {
        "root": str(tmp_path.resolve()),
        "available_phenotypes": ["age", "sex", "diagnosis"],
        "installed_pipelines": FakeNipoppyReader.installed_pipelines,
    }


def test_init_metadata_is_none_when_not_passed(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path)

    assert controller.metadata == {
        "available_phenotypes": None,
        "installed_pipelines": None,
    }


def test_init_metadata_uses_passed_values_as_is(tmp_path, fake_nipoppy_reader):
    # Stored values differing from disk are not checked nor replaced
    controller = NipoppyController(
        root=tmp_path,
        available_phenotypes=["stored_phenotype"],
        installed_pipelines={"processing": {"stored_pipeline": ["1.0"]}},
    )

    assert controller.metadata == {
        "available_phenotypes": ["stored_phenotype"],
        "installed_pipelines": {"processing": {"stored_pipeline": ["1.0"]}},
    }


def test_init_raises_on_dlp(tmp_path, fake_nipoppy_reader):
    with pytest.raises(FedbiomedError, match="not supported"):
        NipoppyController(root=tmp_path, dlp=DataLoadingPlan())


def test_init_raises_on_missing_root(tmp_path, fake_nipoppy_reader):
    with pytest.raises(FedbiomedError, match="does not exist"):
        NipoppyController(root=tmp_path / "missing")


# ---------- registry (dataset registration path) ----------


def test_registry_entry():
    assert REGISTRY_CONTROLLERS[DatasetTypes.NIPOPPY] == (
        NipoppyController,
        NipoppyDataset,
    )


def test_get_controller_with_stored_parameters(tmp_path, fake_nipoppy_reader):
    # As done by the dataset manager, with `dataset_parameters` from the database
    controller = get_controller(
        "nipoppy",
        controller_parameters={
            "root": str(tmp_path),
            "available_phenotypes": ["age"],
            "installed_pipelines": {"processing": {"freesurfer": ["7.3.2"]}},
            "unknown_key": "dropped",
        },
    )

    assert isinstance(controller, NipoppyController)
    assert controller.metadata["available_phenotypes"] == ["age"]


# ---------- select / get_sample ----------


def test_get_sample_before_select_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path)

    with pytest.raises(FedbiomedError, match="Call `select`"):
        controller.get_sample(0)


def test_get_sample_out_of_range_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="baseline")
    controller.select(phenotypes=["sex"])

    with pytest.raises(FedbiomedError, match="index"):
        controller.get_sample(len(controller))
    with pytest.raises(FedbiomedError, match="index"):
        controller.get_sample(-1)


def test_select_reads_requested_columns(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="baseline")
    df = controller.select(phenotypes=["age", "sex"], derivatives=[DERIVATIVE])

    assert fake_nipoppy_reader[0].read_calls == [
        {"phenotypes": ["age", "sex"], "derivatives": [DERIVATIVE]}
    ]
    assert list(df.columns) == ["age", "sex", DERIVATIVE_COLUMN]
    sample = controller.get_sample(0)
    assert list(sample.index) == ["age", "sex", DERIVATIVE_COLUMN]


@pytest.mark.parametrize(
    "session_filters, expected",
    [
        ("baseline", [("sub-01", "baseline"), ("sub-02", "baseline")]),
        (["followup"], [("sub-01", "followup")]),
        (
            [("sub-01", "followup"), ("sub-02", "baseline")],
            [("sub-01", "followup"), ("sub-02", "baseline")],
        ),
        # pairs come back as lists after a JSON round trip
        ([["sub-02", "baseline"]], [("sub-02", "baseline")]),
    ],
)
def test_select_session_filters(tmp_path, fake_nipoppy_reader, session_filters, expected):
    # drop_na removes sub-03 (missing age)
    controller = NipoppyController(root=tmp_path, session_filters=session_filters)
    df = controller.select(phenotypes=["age", "sex"])

    assert sorted(df.index.tolist()) == sorted(expected)


def test_select_invalid_session_filters_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters=[1, 2])

    with pytest.raises(FedbiomedError, match="session_filters"):
        controller.select(phenotypes=["sex"])


def test_select_drop_na(tmp_path, fake_nipoppy_reader):
    kept = NipoppyController(root=tmp_path, session_filters="baseline", drop_na=False)
    assert len(kept.select(phenotypes=["age"])) == 3

    dropped = NipoppyController(root=tmp_path, session_filters="baseline")
    assert len(dropped.select(phenotypes=["age"])) == 2

    # drop_na_kwargs are forwarded to `dropna`
    subset = NipoppyController(
        root=tmp_path, session_filters="baseline", drop_na_kwargs={"subset": ["sex"]}
    )
    assert len(subset.select(phenotypes=["age", "sex"])) == 3


def test_select_applies_whole_df_transform(tmp_path, fake_nipoppy_reader):
    def add_column(df):
        df = df.copy()
        df["age_months"] = df["age"] * 12
        return df

    controller = NipoppyController(
        root=tmp_path, session_filters="baseline", whole_df_transform=add_column
    )
    df = controller.select(phenotypes=["age"])

    assert "age_months" in df.columns
    assert controller.get_sample(0)["age_months"] == 70.0 * 12


def test_select_empty_result_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="missing-session")

    with pytest.raises(FedbiomedError, match="No data left"):
        controller.select(phenotypes=["sex"])


def test_select_multiple_sessions_per_participant_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path)

    with pytest.raises(FedbiomedError, match="more than one session"):
        controller.select(phenotypes=["sex"])


# ---------- shape / len ----------


def test_shape_before_select_without_filters(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, drop_na=False)

    # Number of participant-session rows, as reported by the reader
    assert controller.shape() == {"nipoppy": (4, 1)}
    assert len(controller) == 4


def test_shape_before_select_with_filters(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="baseline")

    # All phenotypes read, filters and dropna applied
    assert controller.shape() == {"nipoppy": (2, 3)}
    assert controller._data is None  # estimation does not select


def test_len_after_select_matches_selected_data(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="baseline")
    df = controller.select(phenotypes=["sex"], derivatives=[DERIVATIVE])

    # sub-03 is kept: its missing `age` is not selected
    assert len(controller) == len(df) == 3
    assert controller.shape() == {"nipoppy": (3, 2)}
