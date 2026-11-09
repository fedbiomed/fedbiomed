import pytest

from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import NipoppyDataset
from fedbiomed.common.dataset._mappings import REGISTRY_CONTROLLERS, get_controller
from fedbiomed.common.dataset_controller import NipoppyController
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from testsupport.fake_nipoppy_reader import DERIVATIVE, DERIVATIVE_COLUMN, FakeNipoppyReader

# ---------- construction ----------


def test_init_controller_kwargs_read_from_disk(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path)

    assert controller._controller_kwargs == {
        "root": str(tmp_path.resolve()),
        "available_phenotypes": ["age", "sex", "diagnosis"],
        "installed_pipelines": FakeNipoppyReader.installed_pipelines,
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


def test_registration_reads_all_phenotypes(tmp_path, fake_nipoppy_reader):
    # As done by the dataset manager at registration: only `root`
    controller = get_controller("nipoppy", controller_parameters={"root": str(tmp_path)})

    assert fake_nipoppy_reader[0].read_calls == [{"phenotypes": None, "derivatives": None}]
    # All participant-session rows, including sub-03 (missing age) and sub-01's
    # second session: neither dropna nor the one-session check apply here
    assert len(controller) == 4
    assert controller.shape() == {"nipoppy": (4, 3)}


def test_get_controller_drops_stored_metadata(tmp_path, fake_nipoppy_reader):
    # `dataset_parameters` from the database contain metadata that is not a
    # constructor argument: it is filtered out by `get_controller`
    controller = get_controller(
        "nipoppy",
        controller_parameters={
            "root": str(tmp_path),
            "available_phenotypes": ["age"],
            "installed_pipelines": {"processing": {"freesurfer": ["7.3.2"]}},
        },
    )

    assert isinstance(controller, NipoppyController)
    assert len(controller) == 4


# ---------- data reading ----------


def test_init_reads_requested_columns(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(
        root=tmp_path,
        phenotypes=["age", "sex"],
        derivatives=[DERIVATIVE],
        session_filters="baseline",
    )

    assert fake_nipoppy_reader[0].read_calls == [
        {"phenotypes": ["age", "sex"], "derivatives": [DERIVATIVE]}
    ]
    sample = controller.get_sample(0)
    assert list(sample.index) == ["age", "sex", DERIVATIVE_COLUMN]


def test_get_sample_out_of_range_raises(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(
        root=tmp_path, phenotypes=["sex"], session_filters="baseline"
    )

    with pytest.raises(FedbiomedError, match="index"):
        controller.get_sample(len(controller))
    with pytest.raises(FedbiomedError, match="index"):
        controller.get_sample(-1)


def test_index(tmp_path, fake_nipoppy_reader):
    controller = NipoppyController(root=tmp_path, session_filters="followup")

    assert controller.index.tolist() == [("sub-01", "followup")]


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
def test_session_filters(tmp_path, fake_nipoppy_reader, session_filters, expected):
    # drop_na removes sub-03 (missing age)
    controller = NipoppyController(
        root=tmp_path,
        phenotypes=["age", "sex"],
        session_filters=session_filters,
        drop_na=True,
    )

    assert sorted(controller.index.tolist()) == sorted(expected)


def test_invalid_session_filters_raises(tmp_path, fake_nipoppy_reader):
    with pytest.raises(FedbiomedError, match="session_filters"):
        NipoppyController(root=tmp_path, phenotypes=["sex"], session_filters=[1, 2])


def test_drop_na(tmp_path, fake_nipoppy_reader):
    # drop_na is off by default
    kept = NipoppyController(root=tmp_path, phenotypes=["age"], session_filters="baseline")
    assert len(kept) == 3

    dropped = NipoppyController(
        root=tmp_path, phenotypes=["age"], session_filters="baseline", drop_na=True
    )
    assert len(dropped) == 2

    # drop_na_kwargs are forwarded to `dropna`
    subset = NipoppyController(
        root=tmp_path,
        phenotypes=["age", "sex"],
        session_filters="baseline",
        drop_na=True,
        drop_na_kwargs={"subset": ["sex"]},
    )
    assert len(subset) == 3


def test_drop_na_depends_on_selected_columns(tmp_path, fake_nipoppy_reader):
    # sub-03 is kept: its missing `age` is not selected
    controller = NipoppyController(
        root=tmp_path,
        phenotypes=["sex"],
        derivatives=[DERIVATIVE],
        session_filters="baseline",
        drop_na=True,
    )

    assert len(controller) == 3
    assert controller.shape() == {"nipoppy": (3, 2)}


def test_applies_whole_df_transform(tmp_path, fake_nipoppy_reader):
    def add_column(df):
        df = df.copy()
        df["age_months"] = df["age"] * 12
        return df

    controller = NipoppyController(
        root=tmp_path,
        phenotypes=["age"],
        session_filters="baseline",
        whole_df_transform=add_column,
    )

    assert controller.get_sample(0)["age_months"] == 70.0 * 12


def test_empty_result_raises(tmp_path, fake_nipoppy_reader):
    with pytest.raises(FedbiomedError, match="No data left"):
        NipoppyController(
            root=tmp_path, phenotypes=["sex"], session_filters="missing-session"
        )


def test_empty_result_does_not_run_whole_df_transform(
    tmp_path, fake_nipoppy_reader, mocker
):
    transform = mocker.Mock()

    with pytest.raises(FedbiomedError, match="No data left"):
        NipoppyController(
            root=tmp_path,
            phenotypes=["sex"],
            session_filters="missing-session",
            whole_df_transform=transform,
        )

    transform.assert_not_called()


def test_wraps_whole_df_transform_errors(tmp_path, fake_nipoppy_reader, mocker):
    error = ValueError("cannot encode column")

    with pytest.raises(FedbiomedError, match="cannot encode column") as exc_info:
        NipoppyController(
            root=tmp_path,
            phenotypes=["sex"],
            session_filters="baseline",
            whole_df_transform=mocker.Mock(side_effect=error),
        )

    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "reader_error,expected_type",
    [
        (ValueError("Must request at least one measure"), FedbiomedValueError),
        (KeyError("unknown phenotype"), FedbiomedError),
        (FileNotFoundError("no derivative file"), FedbiomedError),
    ],
)
def test_wraps_reader_errors(
    tmp_path, fake_nipoppy_reader, mocker, reader_error, expected_type
):
    mocker.patch.object(FakeNipoppyReader, "read", side_effect=reader_error)

    with pytest.raises(FedbiomedError) as exc_info:
        NipoppyController(root=tmp_path, phenotypes=[])

    assert type(exc_info.value) is expected_type
    assert exc_info.value.__cause__ is reader_error
