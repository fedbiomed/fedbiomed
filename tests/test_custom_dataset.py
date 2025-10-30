import pytest

from fedbiomed.common.dataset import CustomDataset
from fedbiomed.common.exceptions import FedbiomedError


class DummyFormat:
    """Simple stub to mimic DataReturnFormat enum item."""

    def __init__(self, py_type):
        self.value = py_type


# ---------- Happy path ----------


def test_good_subclass_initializes_and_gets_items(tmp_path):
    class GoodDS(CustomDataset):
        def read(self):
            # prepare some data once path is set by complete_initialization
            self._data = [([1, 2, 3], [0, 1, 0])]

        def get_item(self, index):
            return self._data[index]

        def __len__(self):
            return len(self._data)

    ds = GoodDS()
    ds.complete_initialization(
        controller_kwargs={"root": str(tmp_path)},
        to_format=DummyFormat(list),
    )

    assert len(ds) == 1
    x, y = ds[0]
    assert isinstance(x, list) and isinstance(y, list)


# ---------- Subclass contract checks (at class creation time) ----------


def test_overriding___getitem___is_forbidden():
    with pytest.raises(FedbiomedError):

        class BadGetItemOverride(CustomDataset):
            def __getitem__(self, idx):  # forbidden
                return super().__getitem__(idx)

            def read(self): ...
            def get_item(self, idx): ...
            def __len__(self):
                return 1


def test_overriding___init___is_forbidden():
    with pytest.raises(FedbiomedError):

        class BadInitOverride(CustomDataset):
            def __init__(self):  # forbidden
                pass

            def read(self): ...
            def get_item(self, idx): ...
            def __len__(self):
                return 1


def test_missing_read_is_forbidden():
    with pytest.raises(FedbiomedError):

        class MissingRead(CustomDataset):
            def get_item(self, idx): ...
            def __len__(self):
                return 0


def test_missing_get_item_is_forbidden():
    with pytest.raises(FedbiomedError):

        class MissingGetItem(CustomDataset):
            def read(self): ...
            def __len__(self):
                return 0


def test_missing___len___is_forbidden():
    with pytest.raises(FedbiomedError):

        class MissingLen(CustomDataset):
            def read(self): ...
            def get_item(self, idx): ...


# ---------- complete_initialization behavior ----------


def test_complete_initialization_requires_root():
    class DS(CustomDataset):
        def read(self): ...
        def get_item(self, idx):
            return ([], [])

        def __len__(self):
            return 1

    ds = DS()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(controller_kwargs={}, to_format=DummyFormat(list))


def test_read_exception_is_wrapped(tmp_path):
    class DS(CustomDataset):
        def read(self):
            raise ValueError("boom")

        def get_item(self, idx):
            return ([], [])

        def __len__(self):
            return 1

    ds = DS()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DummyFormat(list),
        )


def test_get_item_exception_is_wrapped(tmp_path):
    class DS(CustomDataset):
        def read(self): ...
        def get_item(self, idx):
            raise RuntimeError("cannot get")

        def __len__(self):
            return 1

    ds = DS()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DummyFormat(list),
        )


def test_get_item_must_return_tuple_of_len_2(tmp_path):
    class DSWrongTuple(CustomDataset):
        def read(self): ...
        def get_item(self, idx):
            return [1, 2, 3]  # not a (data, target) tuple

        def __len__(self):
            return 1

    ds = DSWrongTuple()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DummyFormat(list),
        )
