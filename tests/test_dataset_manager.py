# tests/test_dataset_manager.py
import pytest

# ---- Import the module under test (adjust the import line if your path differs) ----
# If your project exposes the module at this path, this works as-is:
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager import _dataset_manager as dm_mod

# ---------------------- Lightweight fakes (no sys.modules patching) ----------------------


class FakeTable:
    """In-memory table replacement compatible with DatasetTable/DlpTable/DlbTable/DynamicDatasetTable."""

    def __init__(self, *_, **__):
        self._items = []
        self._by = {}
        self._fake_subtrees = {}

    def insert(self, entry: dict):
        # identify a reasonable key or generate one
        key = None
        for k in ("dataset_id", "dlp_id", "dlb_id"):
            if k in entry and entry[k]:
                key = k
                break
        if key is None:
            key = "dataset_id"
            entry = dict(entry)
            entry[key] = f"auto_{len(self._items) + 1}"
        self._items.append(entry)
        self._by[entry[key]] = entry
        return entry[key]

    def get_by_id(self, key):
        return self._by.get(key)

    def get_all_by_value(self, field, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        s = set(values)
        return [row for row in self._items if row.get(field) in s]

    def delete_by_id(self, key):
        it = self._by.pop(key, None)
        if it is not None:
            self._items = [x for x in self._items if x is not it]

    def all(self):
        return [dict(x) for x in self._items]

    def update_by_id(self, key, updates):
        item = self._by.get(key)
        if item is not None:
            item.update(updates)

    def collect_subtree(self, parent_id):
        return self._fake_subtrees.get(parent_id, [parent_id])


class FakeController:
    def __init__(self, **kwargs):
        self._controller_kwargs = kwargs

    def shape(self):
        return (3, 4)

    def get_types(self):
        return {"col1": "float", "col2": "int"}


class MinimalDLP:
    """Drop-in minimal DLP used by tests. Replace with your real DataLoadingPlan if desired."""

    def __init__(self, dlp_id="dlp-1", loading_blocks=None):
        self.dlp_id = dlp_id
        self.loading_blocks = loading_blocks or {"lb1": "dlb-1", "lb2": "dlb-2"}

    def serialize(self):
        dlp_meta = {"dlp_id": self.dlp_id, "loading_blocks": self.loading_blocks}
        dlbs_meta = [{"dlb_id": v} for v in self.loading_blocks.values()]
        return dlp_meta, dlbs_meta


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Replace the concrete table classes and the controller getter *on the module under test*
    so we don't need to mess with sys.modules. Each test starts with a clean in-memory DB.
    """
    monkeypatch.setattr(dm_mod, "DatasetTable", FakeTable, raising=True)
    monkeypatch.setattr(dm_mod, "DynamicDatasetTable", FakeTable, raising=True)
    monkeypatch.setattr(dm_mod, "DlpTable", FakeTable, raising=True)
    monkeypatch.setattr(dm_mod, "DlbTable", FakeTable, raising=True)

    # If the module imported the factory as a name (e.g., `from ... import get_controller`),
    # patch that name; if it calls via a module attribute, patch that as well if present.
    def fake_get_controller(data_type, controller_parameters):
        return FakeController(**controller_parameters)

    # Try both common spots; only patch those that exist to keep it robust.
    if hasattr(dm_mod, "get_controller"):
        monkeypatch.setattr(dm_mod, "get_controller", fake_get_controller, raising=True)
    if hasattr(dm_mod, "_registry_controllers"):
        monkeypatch.setattr(
            dm_mod._registry_controllers,
            "get_controller",
            fake_get_controller,
            raising=True,
        )

    yield
    # teardown handled by monkeypatch


# ------------------------------------- TESTS -------------------------------------


def test_read_csv_header_and_semicolon(tmp_path):
    dm = dm_mod.DatasetManager(path="/tmp/db")

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


def test_save_data_loading_plan_and_get_back():
    dm = dm_mod.DatasetManager(path="/db")
    dlp = MinimalDLP(dlp_id="DLP-1", loading_blocks={"x": "DLB-9"})

    rid = dm.save_data_loading_plan(dlp)
    assert rid == "DLP-1"

    got_dlp, dlbs = dm.get_dlp_by_id("DLP-1")
    assert got_dlp and got_dlp["dlp_id"] == "DLP-1"
    assert {d["dlb_id"] for d in dlbs} == {"DLB-9"}


def test_add_database_with_and_without_persisting_dlp():
    dm = dm_mod.DatasetManager(path="/db")

    dlp_keep = MinimalDLP(dlp_id="keep-me", loading_blocks={"a": "A", "b": "B"})
    did1 = dm.add_database(
        name="N",
        data_type="tabular",
        tags=["t"],
        description="d",
        path="/root",
        dataset_parameters={"alpha": 1},
        data_loading_plan=dlp_keep,
        save_dlp=True,
    )
    ds1 = dm.dataset_table.get_by_id(did1)
    assert ds1["name"] == "N"
    assert ds1["shape"] == (3, 4)
    assert dm.dlp_table.get_by_id("keep-me") is not None

    dlp_skip = MinimalDLP(dlp_id="dont-save")
    did2 = dm.add_database(
        name="M",
        data_type="tabular",
        tags=[],
        description="",
        path="/r",
        dataset_parameters={},
        data_loading_plan=dlp_skip,
        save_dlp=False,
    )
    assert dm.dlp_table.get_by_id("dont-save") is None
    ds2 = dm.dataset_table.get_by_id(did2)
    assert ds2["dlp_id"] == "dont-save"


def test_remove_dlp_by_id_existing_and_missing():
    dm = dm_mod.DatasetManager(path="/db")
    dm.dlp_table.insert({"dlp_id": "X", "loading_blocks": {"a": "d1", "b": "d2"}})
    dm.dlb_table.insert({"dlb_id": "d1"})
    dm.dlb_table.insert({"dlb_id": "d2"})

    dm.remove_dlp_by_id("X")
    assert dm.dlp_table.get_by_id("X") is None
    assert dm.dlb_table.get_by_id("d1") is None
    assert dm.dlb_table.get_by_id("d2") is None

    # Removing a non-existent DLP should be a no-op (no exception)
    dm.remove_dlp_by_id("does-not-exist")


def test_list_my_datasets_strips_dtypes_and_prints(capsys):
    dm = dm_mod.DatasetManager(path="/db")
    dm.dataset_table.insert({"dataset_id": "ds1", "name": "A", "dtypes": {"x": "int"}})
    out = dm.list_my_datasets(verbose=True)

    assert isinstance(out, list) and out
    assert "dtypes" not in out[0]  # the method should remove dtypes before returning
    assert capsys.readouterr().out.strip() != ""  # something was printed


def test_save_data_loading_plan_none_returns_none():
    dm = dm_mod.DatasetManager(path="/db")
    assert dm.save_data_loading_plan(None) is None


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


def test_get_dataset_entry_from_both_tables():
    dm = dm_mod.DatasetManager(path="/db")

    # Insert raw dataset
    raw_id = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

    # Insert dynamic dataset
    dyn_id = dm.dynamic_dataset_table.insert(
        {"dataset_id": "dyn1", "parent_dataset_id": "raw1"}
    )

    assert dm.get_dataset_entry_by_id("raw1")["dataset_id"] == raw_id
    assert dm.get_dataset_entry_by_id("dyn1")["dataset_id"] == dyn_id

    with pytest.raises(FedbiomedError):
        dm.get_dataset_entry_by_id("does-not-exist")


def test_add_dynamic_dataset():
    dm = dm_mod.DatasetManager(path="/db")

    # Insert parent raw dataset
    parent_id = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

    dyn_id = dm.add_dynamic_dataset(
        path="/path/to/dynamic",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id=parent_id,
    )

    # DB entry exists
    entry = dm.dynamic_dataset_table.get_by_id(dyn_id)
    assert entry is not None
    assert entry["path"] == "/path/to/dynamic"
    assert entry["data_type"] == "tabular"
    assert entry["shape"] == (3, 4)
    assert entry["dtypes"] == {"col1": "float", "col2": "int"}


def test_delete_dynamic_single():
    dm = dm_mod.DatasetManager(path="/db")

    parent_id = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

    dyn_id = dm.add_dynamic_dataset(
        path="/path/to/dynamic",
        researcher_id="r",
        experiment_id="e",
        processing_id="p",
        parent_dataset_id=parent_id,
    )

    dm.delete_dataset_by_id(dyn_id)

    assert dm.dynamic_dataset_table.get_by_id(dyn_id) is None


def test_delete_dynamic_dataset_recursive():
    dm = dm_mod.DatasetManager(path="/db")

    parent = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

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

    # Inject fake subtree behavior
    dm.dynamic_dataset_table._fake_subtrees = {child1: [child1, child2]}

    dm.delete_dataset_by_id(child1, recursive=True)

    assert dm.dataset_table.get_by_id(parent) is not None
    assert dm.dynamic_dataset_table.get_by_id(child1) is None
    assert dm.dynamic_dataset_table.get_by_id(child2) is None


def test_delete_dynamic_dataset_with_force_reassigns_children():
    dm = dm_mod.DatasetManager(path="/db")

    parent = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

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

    dm.delete_dataset_by_id(child1, force=True)

    # child2 should now point to raw parent
    updated = dm.dynamic_dataset_table.get_by_id(child2)
    assert updated["parent_dataset_id"] == parent

    assert dm.dynamic_dataset_table.get_by_id(child1) is None


def test_delete_dataset_by_id_raises():
    dm = dm_mod.DatasetManager(path="/db")

    parent = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

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
    )

    with pytest.raises(FedbiomedError):
        dm.delete_dataset_by_id(child1)
    with pytest.raises(FedbiomedError):
        dm.delete_dataset_by_id(parent)
    with pytest.raises(FedbiomedError):
        dm.delete_dataset_by_id(parent, force=True)


def test_delete_dataset_without_children():
    dm = dm_mod.DatasetManager(path="/db")

    parent1 = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})
    parent2 = dm.dataset_table.insert({"dataset_id": "raw2", "data_type": "tabular"})

    _ = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent1,
    )

    dm.delete_dataset_by_id(parent2)

    assert dm.dataset_table.get_by_id(parent2) is None
    assert dm.dataset_table.get_by_id(parent1) is not None


def test_delete_dataset_with_children_recursive():
    dm = dm_mod.DatasetManager(path="/db")

    parent = dm.dataset_table.insert({"dataset_id": "raw1", "data_type": "tabular"})

    child = dm.add_dynamic_dataset(
        path="/path/to/dynamic/child1",
        researcher_id="r",
        experiment_id="e",
        processing_id="p1",
        parent_dataset_id=parent,
    )

    # Inject fake subtree behavior
    dm.dynamic_dataset_table._fake_subtrees = {parent: [parent, child]}

    dm.delete_dataset_by_id(parent, recursive=True)

    assert dm.dataset_table.get_by_id(parent) is None
    assert dm.dynamic_dataset_table.get_by_id(child) is None
