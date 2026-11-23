# tests/test_dataset_manager.py
import pytest
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager import _dataset_manager as dm_mod
from fedbiomed.node.db_models import DatasetEntry, DlbEntry
from sqlalchemy import select

# ---------------------- Lightweight fakes (no sys.modules patching) ----------------------


class FakeController:
    def __init__(self, n_samples=100, **kwargs):
        self._controller_kwargs = kwargs
        self._n_samples = n_samples

    def __len__(self):
        return self._n_samples

    def shape(self):
        return {"csv": (3, 4)}

    def get_types(self):
        return {"col1": "float", "col2": "int"}


class MinimalDLP:
    """Drop-in minimal DLP used by tests."""

    def __init__(self, dlp_id="dlp-1", loading_blocks=None, dlp_name=None):
        self.dlp_id = dlp_id
        self.dlp_name = dlp_name or f"plan {dlp_id}"
        self.loading_blocks = loading_blocks or {"lb1": "dlb-1", "lb2": "dlb-2"}

    def serialize(self):
        # Same shape as DataLoadingPlan.serialize: key_paths values are tuples
        dlp_meta = {
            "dlp_id": self.dlp_id,
            "dlp_name": self.dlp_name,
            "target_dataset_type": "csv",
            "loading_blocks": self.loading_blocks,
            "key_paths": {key: ("module", "Class") for key in self.loading_blocks},
        }
        dlbs_meta = [
            {"dlb_id": v, "loading_block_class": "Block", "loading_block_module": "m"}
            for v in self.loading_blocks.values()
        ]
        return dlp_meta, dlbs_meta


@pytest.fixture
def db_path(tmp_path):
    """Path of a node database in `tmp_path`, with the manager's tables."""
    db_path = str(tmp_path / "node.json")
    get_database(db_path).create_tables(dm_mod.DatasetManager.MODELS)
    return db_path


def _add_dataset(db_path, dataset_id, **fields):
    """Registers a dataset directly in the database, without loading it."""
    with get_database(db_path).transaction() as session:
        session.add(
            DatasetEntry.from_dict(
                {
                    "dataset_id": dataset_id,
                    "name": dataset_id,
                    "data_type": "csv",
                    "tags": [dataset_id],
                    "description": "d",
                    "path": "/data",
                    **fields,
                }
            )
        )
    return dataset_id


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Replaces the controller getter so tests do not load real datasets."""

    def fake_get_controller(data_type, controller_parameters):
        return FakeController(**controller_parameters)

    monkeypatch.setattr(dm_mod, "get_controller", fake_get_controller)


# ------------------------------------- TESTS -------------------------------------


def test_read_csv_header_and_semicolon(tmp_path, db_path):
    dm = dm_mod.DatasetManager(db_path)

    f_header = tmp_path / "with_header.csv"
    f_header.write_text("c1,c2\n1,2\n3,4\n", encoding="utf-8")
    df1 = dm.read_csv(str(f_header))
    assert list(df1.columns) == ["c1", "c2"]
    assert df1.iloc[0, 0] == 1

    f_sc = tmp_path / "semicolon.csv"
    f_sc.write_text("1;2\n3;4\n", encoding="utf-8")
    df2 = dm.read_csv(str(f_sc))
    # with no header, pandas will number columns 0,1
    assert list(df2.columns) == [0, 1]
    assert df2.iloc[1, 0] == 3


def test_save_data_loading_plan_and_get_back(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dlp = MinimalDLP(dlp_id="DLP-1", loading_blocks={"x": "DLB-9"})

    rid = dm.save_data_loading_plan(dlp)
    assert rid == "DLP-1"

    got_dlp, dlbs = dm.get_dlp_by_id("DLP-1")
    assert got_dlp and got_dlp["dlp_id"] == "DLP-1"
    assert got_dlp["key_paths"] == {"x": ["module", "Class"]}
    assert {d["dlb_id"] for d in dlbs} == {"DLB-9"}


def test_get_dlp_by_id_not_found(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dlp, dlbs = dm.get_dlp_by_id("does-not-exist")
    assert dlp is None
    assert dlbs == []


@pytest.mark.parametrize(
    "dataset_parameters,expected",
    [
        (None, ["/data"]),
        ({"download": True}, ["/data"]),
        # Pointing at files already in place deploys without a fetch
        ({"download": False}, []),
    ],
)
def test_add_database_deploys_mnist_files(
    monkeypatch, db_path, dataset_parameters, expected
):
    """Deploying MNIST fetches its files unless the caller says they are there."""
    fetched = []
    monkeypatch.setattr(dm_mod, "download_mnist", fetched.append)
    dm = dm_mod.DatasetManager(db_path)

    dm.add_database(
        name="MNIST",
        data_type="default",
        tags=["#MNIST"],
        description="d",
        path="/data",
        dataset_parameters=dataset_parameters,
    )

    assert fetched == expected


def test_add_database_does_not_deploy_other_types(monkeypatch, db_path):
    """Only MNIST can be fetched; every other type must already be on disk."""
    fetched = []
    monkeypatch.setattr(dm_mod, "download_mnist", fetched.append)
    dm = dm_mod.DatasetManager(db_path)

    dm.add_database(
        name="N", data_type="csv", tags=["t"], description="d", path="/root"
    )

    assert fetched == []


def test_add_database_refers_to_saved_dlp(db_path):
    dm = dm_mod.DatasetManager(db_path)

    dlp_keep = MinimalDLP(dlp_id="keep-me", loading_blocks={"a": "A", "b": "B"})
    dm.save_data_loading_plan(dlp_keep)
    did1 = dm.add_database(
        name="N",
        data_type="csv",
        tags=["t"],
        description="d",
        path="/root",
        dataset_parameters={"alpha": 1},
        data_loading_plan=dlp_keep,
    )
    ds1 = dm.get_dataset(did1)
    assert ds1["name"] == "N"
    assert ds1["shape"] == {"csv": [3, 4]}
    assert dm.get_dlp_by_id("keep-me")[0] is not None

    # A new dataset using the same plan on purpose refers to it
    did2 = dm.add_database(
        name="M",
        data_type="csv",
        tags=["u"],
        description="",
        path="/r",
        dataset_parameters={},
        data_loading_plan=dlp_keep,
    )
    ds2 = dm.get_dataset(did2)
    assert ds2["dlp_id"] == "keep-me"
    assert len(dm.get_dlp_by_id("keep-me")[1]) == 2


def test_add_database_refuses_unsaved_dlp(db_path):
    dm = dm_mod.DatasetManager(db_path)

    with pytest.raises(FedbiomedError, match="FB634.*FOREIGN KEY constraint failed"):
        dm.add_database(
            name="N",
            data_type="csv",
            tags=["t"],
            description="d",
            path="/root",
            data_loading_plan=MinimalDLP(dlp_id="unsaved"),
        )
    assert dm.list_my_datasets(verbose=False) == []


def test_remove_dlp_by_id_existing_and_missing(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dm.save_data_loading_plan(
        MinimalDLP(dlp_id="X", loading_blocks={"a": "d1", "b": "d2"})
    )

    dm.remove_dlp_by_id("X")
    assert dm.get_dlp_by_id("X") == (None, [])
    with get_database(db_path).transaction() as session:
        assert session.scalars(select(DlbEntry)).all() == []

    # Removing a non-existent DLP should be a no-op (no exception)
    dm.remove_dlp_by_id("does-not-exist")


def test_list_my_datasets_strips_dtypes_and_prints(capsys, db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "ds1", name="A", dtypes={"x": "int"})
    out = dm.list_my_datasets(verbose=True)

    assert isinstance(out, list) and out
    assert "dtypes" not in out[0]  # the method should remove dtypes before returning
    assert capsys.readouterr().out.strip() != ""  # something was printed


def test_list_my_datasets_not_verbose(capsys, db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "ds1", name="B", dtypes={"x": "int"})
    out = dm.list_my_datasets(verbose=False)

    assert isinstance(out, list) and out
    assert "dtypes" not in out[0]
    assert capsys.readouterr().out == ""  # nothing printed


def test_obfuscate_private_information_hides_sensitive_fields():
    data = [
        {
            "path": "/secret",
            "data_type": "medical-folder",
            "dataset_parameters": {"tabular_file": "a.csv", "k": 1},
        },
        {
            "path": "/public",
            "data_type": "image-folder",
            "dataset_parameters": {"z": 2},
        },
    ]
    obf = list(dm_mod.DatasetManager.obfuscate_private_information(data))
    assert "path" not in obf[0] and "path" not in obf[1]
    # medical-folder should drop 'tabular_file'
    assert "tabular_file" not in obf[0]["dataset_parameters"]
    # other dataset types keep their parameters
    assert obf[1]["dataset_parameters"]["z"] == 2


def test_obfuscate_private_information_raises_on_bad_item():
    with pytest.raises(FedbiomedError):
        list(dm_mod.DatasetManager.obfuscate_private_information([{"path": "x"}, 123]))


def test_get_dataset_by_scope(db_path):
    dm = dm_mod.DatasetManager(db_path)

    # Insert raw dataset
    raw_id = _add_dataset(db_path, "raw1")

    # Insert dynamic dataset
    dyn_id = dm.add_dynamic_dataset(
        path="/dyn",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id="raw1",
        dataset_id="dyn1",
    )

    assert dm.get_dataset("raw1")["dataset_id"] == raw_id
    assert dm.get_dataset("dyn1") is None
    assert dm.get_dataset("dyn1", scope="dynamic")["dataset_id"] == dyn_id
    assert dm.get_dataset("raw1", scope="dynamic") is None
    assert dm.get_dataset("raw1", scope="both")["dataset_id"] == raw_id
    assert dm.get_dataset("dyn1", scope="both")["dataset_id"] == dyn_id
    assert dm.get_dataset("does-not-exist", scope="both") is None
    with pytest.raises(FedbiomedError, match="FB632.*original"):
        dm.get_dataset("raw1", scope="everything")


def test_add_dynamic_dataset(db_path):
    dm = dm_mod.DatasetManager(db_path)

    # Insert parent raw dataset
    parent_id = _add_dataset(db_path, "raw1")

    dyn_id = dm.add_dynamic_dataset(
        path="/path/to/dynamic",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id=parent_id,
    )

    # DB entry exists
    entry = dm.get_dataset(dyn_id, scope="dynamic")
    assert entry is not None
    assert entry["path"] == "/path/to/dynamic"
    assert entry["data_type"] == "csv"
    assert entry["shape"] == {"csv": [3, 4]}
    assert entry["dtypes"] == {"col1": "float", "col2": "int"}
    assert entry["parent_dataset_id"] == parent_id
    assert "parent_dynamic_id" not in entry

    # Derived from a dynamic dataset: the other parent column is set
    grandchild = dm.get_dataset(
        dm.add_dynamic_dataset(
            path="/path/to/grandchild",
            researcher_id="r",
            experiment_id="e",
            processing_id="p2",
            parent_dataset_id=dyn_id,
        ),
        "dynamic",
    )
    assert grandchild["parent_dynamic_id"] == dyn_id
    assert "parent_dataset_id" not in grandchild
    assert grandchild["data_type"] == "csv"


def test_add_dynamic_dataset_parent_not_found(db_path):
    dm = dm_mod.DatasetManager(db_path)
    with pytest.raises(FedbiomedError):
        dm.add_dynamic_dataset(
            path="/path",
            researcher_id="r",
            experiment_id="e",
            processing_id="p",
            parent_dataset_id="does-not-exist",
        )


def test_build_dataset_entry_strips_root_from_dataset_parameters(db_path):
    """root (and dlp) must not leak into the stored dataset_parameters."""
    dm = dm_mod.DatasetManager(db_path)
    did = dm.add_database(
        name="N",
        data_type="csv",
        tags=["t"],
        description="d",
        path="/root",
        dataset_parameters={"extra_param": 42},
    )
    entry = dm.get_dataset(did)
    assert "root" not in entry["dataset_parameters"]
    assert entry["dataset_parameters"].get("extra_param") == 42


def test_delete_dynamic_single(db_path):
    dm = dm_mod.DatasetManager(db_path)

    parent_id = _add_dataset(db_path, "raw1")

    dyn_id = dm.add_dynamic_dataset(
        path="/path/to/dynamic",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id=parent_id,
    )

    dm.delete_dataset_by_id(dyn_id)

    assert dm.get_dataset(dyn_id, scope="dynamic") is None


def test_delete_dynamic_dataset_recursive(db_path):
    dm = dm_mod.DatasetManager(db_path)

    parent = _add_dataset(db_path, "raw1")

    child1 = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent,
    )
    child2 = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child2",
        researcher_id="r",
        experiment_id="e",
        processing_id="p2",
        parent_dataset_id=child1,
    )
    child3 = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child3",
        researcher_id="r",
        experiment_id="e",
        processing_id="p3",
        parent_dataset_id=child2,
    )
    child4 = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child4",
        researcher_id="r",
        experiment_id="e",
        processing_id="p4",
        parent_dataset_id=parent,
    )

    dm.delete_dataset_by_id(child1, recursive=True)
    # Test recursive works even with no children
    dm.delete_dataset_by_id(child4, recursive=True)

    assert dm.get_dataset(parent) is not None
    for child in (child1, child2, child3, child4):
        assert dm.get_dataset(child, scope="dynamic") is None


def test_delete_dataset_by_id_raises(db_path):
    dm = dm_mod.DatasetManager(db_path)

    parent = _add_dataset(db_path, "raw1")

    child1 = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent,
    )
    _ = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child2",
        researcher_id="r",
        experiment_id="e",
        processing_id="p2",
        parent_dataset_id=child1,
        name="second",
    )

    with pytest.raises(
        FedbiomedError,
        match=rf"FB632: .*Dataset '{child1}' has derived datasets: 'second'\. "
        "Delete them too with recursive=True",
    ):
        dm.delete_dataset_by_id(child1)
    with pytest.raises(
        FedbiomedError, match="Dataset 'raw1' has derived datasets: "
    ) as e:
        dm.delete_dataset_by_id(parent)
    assert f"'{child1}'" in str(e.value) and "'second'" in str(e.value)
    assert dm.get_dataset(parent) is not None


def test_delete_dataset_without_children(db_path):
    dm = dm_mod.DatasetManager(db_path)

    parent1 = _add_dataset(db_path, "raw1")
    parent2 = _add_dataset(db_path, "raw2")

    _ = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent1,
    )

    dm.delete_dataset_by_id(parent2)

    assert dm.get_dataset(parent2) is None
    assert dm.get_dataset(parent1) is not None


def test_delete_dataset_with_children_recursive(db_path):
    dm = dm_mod.DatasetManager(db_path)

    parent = _add_dataset(db_path, "raw1")

    child = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent,
    )

    dm.delete_dataset_by_id(parent, recursive=True)

    assert dm.get_dataset(parent) is None
    assert dm.get_dataset(child, scope="dynamic") is None


# ------------------------------------- MINIMUM SAMPLES TESTS -------------------------------------


@pytest.mark.parametrize(
    "min_samples,n_samples",
    [
        (0, 0),  # default: no minimum, zero samples allowed
        (0, 100),  # default: no minimum, any size accepted
        (50, 100),  # above minimum
        (50, 50),  # exactly at minimum (boundary)
    ],
)
def test_validate_samples_does_not_raise(db_path, min_samples, n_samples):
    dm = dm_mod.DatasetManager(db_path, min_samples=min_samples)
    dm.validate_samples(n_samples)  # must not raise


def test_validate_samples_below_minimum_raises(db_path):
    """validate_samples raises FedbiomedError when n_samples < min_samples."""
    dm = dm_mod.DatasetManager(db_path, min_samples=50)
    with pytest.raises(FedbiomedError, match="below the node's minimum required"):
        dm.validate_samples(10)


def test_add_database_below_minimum_samples_raises(monkeypatch, db_path):
    """add_database raises FedbiomedError via validate_samples when dataset is too small."""
    monkeypatch.setattr(
        dm_mod,
        "get_controller",
        lambda data_type, controller_parameters: FakeController(
            n_samples=10, **controller_parameters
        ),
    )
    dm = dm_mod.DatasetManager(db_path, min_samples=50)
    with pytest.raises(FedbiomedError, match="below the node's minimum required"):
        dm.add_database(
            name="small",
            data_type="csv",
            tags=["t"],
            description="d",
            path="/root",
        )


def test_add_dynamic_dataset_below_minimum_samples_raises(monkeypatch, db_path):
    """add_dynamic_dataset also enforces min_samples via _build_dataset_entry."""
    monkeypatch.setattr(
        dm_mod,
        "get_controller",
        lambda data_type, controller_parameters: FakeController(
            n_samples=5, **controller_parameters
        ),
    )
    dm = dm_mod.DatasetManager(db_path, min_samples=50)
    _add_dataset(db_path, "raw1")
    with pytest.raises(FedbiomedError, match="below the node's minimum required"):
        dm.add_dynamic_dataset(
            path="/path",
            researcher_id="r",
            experiment_id="e",
            processing_id="p",
            parent_dataset_id="raw1",
        )


def test_custom_data_type_skips_sample_validation(monkeypatch, db_path):
    """Custom datasets bypass min_samples since they don't expose a fixed sample count."""
    monkeypatch.setattr(
        dm_mod,
        "get_controller",
        lambda data_type, controller_parameters: FakeController(
            n_samples=0, **controller_parameters
        ),
    )
    dm = dm_mod.DatasetManager(db_path, min_samples=100)
    # Must not raise despite n_samples=0 < min_samples=100
    dm.add_database(name="C", data_type="custom", tags=[], description="", path="/root")


def test_build_dataset_entry_strips_dlp_from_dataset_parameters(db_path):
    """The 'dlp' controller kwarg must not appear in the stored dataset_parameters."""
    dm = dm_mod.DatasetManager(db_path)
    dlp = MinimalDLP(dlp_id="dlp-x")
    dm.save_data_loading_plan(dlp)
    did = dm.add_database(
        name="N",
        data_type="csv",
        tags=[],
        description="",
        path="/root",
        data_loading_plan=dlp,
    )
    entry = dm.get_dataset(did)
    assert "dlp" not in entry["dataset_parameters"]
    assert entry["dlp_id"] == "dlp-x"


def test_read_csv_with_index_col(tmp_path, db_path):
    dm = dm_mod.DatasetManager(db_path)
    f = tmp_path / "indexed.csv"
    f.write_text("idx,val\na,1\nb,2\n", encoding="utf-8")
    df = dm.read_csv(str(f), index_col=0)
    assert df.index.tolist() == ["a", "b"]
    assert list(df.columns) == ["val"]


def test_read_csv_with_dtype_str_preserves_leading_zeros(tmp_path, db_path):
    """`dtype=str` keeps numeric-looking values (e.g. folder-name references)
    as-is instead of letting pandas infer them as integers and drop leading
    zeros. Used by the GUI to preview MedicalFolderDataset reference CSVs.
    """
    dm = dm_mod.DatasetManager(db_path)
    f = tmp_path / "folder_names.csv"
    f.write_text("folder_name,age\n0002,30\n1005,40\n", encoding="utf-8")

    df_inferred = dm.read_csv(str(f))
    assert df_inferred["folder_name"].tolist() == [2, 1005]

    df_str = dm.read_csv(str(f), dtype=str)
    assert df_str["folder_name"].tolist() == ["0002", "1005"]
    assert df_str["age"].tolist() == ["30", "40"]


def test_add_dynamic_dataset_with_optional_fields(db_path):
    """All optional fields are forwarded and stored correctly."""
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "raw1")

    dyn_id = dm.add_dynamic_dataset(
        path="/path",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id="raw1",
        name="my-dyn",
        tags=["tag1"],
        description="some desc",
        dataset_id="explicit-dyn-id",
        dataset_parameters={"extra": 42},
    )

    assert dyn_id == "explicit-dyn-id"
    entry = dm.get_dataset(dyn_id, scope="dynamic")
    assert entry["name"] == "my-dyn"
    assert entry["tags"] == ["tag1"]
    assert entry["description"] == "some desc"
    assert entry["dataset_parameters"].get("extra") == 42


def test_delete_dataset_by_id_not_found(db_path):
    dm = dm_mod.DatasetManager(db_path)
    with pytest.raises(FedbiomedError):
        dm.delete_dataset_by_id("no-such-id")


# ------------------------------------- DATABASE RULES -------------------------------------


def test_add_database_refuses_conflicting_tags_and_saves_nothing(db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "raw1", name="First", tags=["a", "b"])

    with pytest.raises(FedbiomedError, match="FB322.*Conflicting dataset names: First"):
        dm.add_database(
            name="N",
            data_type="csv",
            tags=["a"],
            description="d",
            path="/root",
        )


def test_save_data_loading_plan_refuses_saved_block(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dm.save_data_loading_plan(MinimalDLP(dlp_id="P1", loading_blocks={"a": "A"}))

    with pytest.raises(
        FedbiomedError, match="FB634.*UNIQUE constraint failed: Dlbs.dlb_id"
    ):
        dm.save_data_loading_plan(MinimalDLP(dlp_id="P2", loading_blocks={"a": "A"}))
    assert dm.get_dlp_by_id("P2") == (None, [])


def test_save_data_loading_plan_refuses_name_used_by_another_plan(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dm.save_data_loading_plan(MinimalDLP(dlp_id="P1", dlp_name="same name"))

    with pytest.raises(FedbiomedError, match="FB316.*DLP name needs to be unique"):
        dm.save_data_loading_plan(MinimalDLP(dlp_id="P2", dlp_name="same name"))


def test_remove_dlp_by_id_refuses_plan_in_use(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dm.save_data_loading_plan(MinimalDLP(dlp_id="P"))
    dm.add_database(
        name="User",
        data_type="csv",
        tags=["t"],
        description="d",
        path="/root",
        data_loading_plan=MinimalDLP(dlp_id="P"),
    )

    with pytest.raises(FedbiomedError, match="FB316.*used by datasets: User"):
        dm.remove_dlp_by_id("P")
    assert dm.get_dlp_by_id("P")[0] is not None


def test_search_datasets_matches_name_or_description_once(db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "d1", name="ÉTUDE brain", description="étude scans")
    _add_dataset(db_path, "d2", name="Other", description="Brain 50%")
    _add_dataset(db_path, "d3", name="Unrelated", description="x")

    found = dm.search_datasets("étude")
    assert [d["dataset_id"] for d in found] == ["d1"]
    assert "dtypes" not in found[0]
    assert {d["dataset_id"] for d in dm.search_datasets("brain")} == {"d1", "d2"}
    assert [d["dataset_id"] for d in dm.search_datasets("50%")] == ["d2"]


def test_search_by_tags_and_conflicting_tags(db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "d1", tags=["a", "b"])
    _add_dataset(db_path, "d2", tags=["c"])

    assert [d["dataset_id"] for d in dm.search_by_tags(["a"])] == ["d1"]
    assert dm.search_by_tags(["a", "c"]) == []
    conflicting = dm.search_conflicting_tags(["a", "b", "x"])
    assert [d["dataset_id"] for d in conflicting] == ["d1"]
    assert dm.search_conflicting_tags(["z"]) == []


def test_update_dataset_returns_updated_entry(db_path):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "d1", tags=["a"])
    _add_dataset(db_path, "d2", name="Second", tags=["b"])

    updated = dm.update_dataset("d1", {"name": "new", "tags": ["a", "x"]})

    assert updated["name"] == "new" and updated["tags"] == ["a", "x"]
    assert dm.get_dataset("d1")["name"] == "new"


@pytest.mark.parametrize(
    "dataset_id, changes, match",
    [
        ("missing", {"name": "x"}, "FB632.*not found"),
        ("d1", {"dataset_id": "other"}, "FB632.*Cannot change the field"),
        ("d1", {"colour": "red"}, r"FB632.*unknown fields \['colour'\]"),
        ("d1", {"tags": ["b", "c"]}, "FB322.*Conflicting dataset names: Second"),
        ("d1", {"data_type": "tabular"}, "FB634.*CHECK constraint failed"),
    ],
)
def test_update_dataset_refuses_invalid_changes(db_path, dataset_id, changes, match):
    dm = dm_mod.DatasetManager(db_path)
    _add_dataset(db_path, "d1", tags=["a"])
    _add_dataset(db_path, "d2", name="Second", tags=["b"])

    with pytest.raises(FedbiomedError, match=match):
        dm.update_dataset(dataset_id, changes)

    assert dm.get_dataset("d1")["tags"] == ["a"]


def test_list_dlps_by_target_dataset_type(db_path):
    dm = dm_mod.DatasetManager(db_path)
    dm.save_data_loading_plan(MinimalDLP(dlp_id="P"))

    assert [d["dlp_id"] for d in dm.list_dlps("csv")] == ["P"]
    assert dm.list_dlps("images") == []
    with pytest.raises(FedbiomedError, match="FB632.*target_dataset_type"):
        dm.list_dlps("tabular")
