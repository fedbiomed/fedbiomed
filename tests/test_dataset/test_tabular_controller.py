from pathlib import Path
from typing import Union

import pytest

from fedbiomed.common.dataset_controller import TabularController
from fedbiomed.common.exceptions import FedbiomedError

# ---------- init / wiring ----------


def test_init_passes_root_and_columns_to_reader(mocker, tmp_path):
    # Patch CsvReader where it's used
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )
    reader_instance = CsvReaderMock.return_value

    controller = TabularController(root=tmp_path)

    # CsvReader called once; arg equals controller.root (avoid str/Path brittleness)
    CsvReaderMock.assert_called_once()
    called_arg = CsvReaderMock.call_args.args[0]
    assert called_arg == controller.root

    # Attributes set as provided (root is not coerced; stays Path here)
    assert controller.root == tmp_path

    # Reader stored
    assert controller._reader is reader_instance


def test_init_with_string_root(mocker, tmp_path):
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )

    controller = TabularController(root=str(tmp_path))

    CsvReaderMock.assert_called_once()
    called_arg = CsvReaderMock.call_args.args[0]
    assert called_arg == controller.root  # matches whatever the controller stored
    assert isinstance(controller.root, Union[str, Path])
    assert str(controller.root) == str(tmp_path)


def test_init_propagates_reader_error(mocker):
    # If the underlying CsvReader rejects the path, controller should propagate FedbiomedError
    mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader",
        side_effect=FedbiomedError("boom"),
    )
    with pytest.raises(FedbiomedError):
        TabularController(root="/does/not/exist")


# ---------- __len__ / shape delegation ----------


def test_len_delegates_to_reader(mocker, tmp_path):
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )
    CsvReaderMock.return_value.len.return_value = 123

    controller = TabularController(tmp_path)

    assert len(controller) == 123
    CsvReaderMock.return_value.len.assert_called_once_with()


def test_shape_delegates_to_reader(mocker, tmp_path):
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )
    CsvReaderMock.return_value.shape.return_value = {"rows": 10, "cols": 3}

    controller = TabularController(tmp_path)

    assert controller.shape() == {"rows": 10, "cols": 3}
    CsvReaderMock.return_value.shape.assert_called_once_with()


# ---------- get_sample behavior ----------


def testget_sample_happy_path(mocker, tmp_path):
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )
    reader = CsvReaderMock.return_value
    # Pretend we have 5 rows
    reader.len.return_value = 5

    # Simulate reader.get for data and target
    reader.get.side_effect = [
        [[1, 2, 3, 4]],  # input_columns
    ]

    controller = TabularController(tmp_path)

    result = controller.get_sample(3)
    assert result == [[1, 2, 3, 4]]


def testget_sample_out_of_range_raises(mocker, tmp_path):
    CsvReaderMock = mocker.patch(
        "fedbiomed.common.dataset_controller._tabular_controller.CsvReader"
    )
    reader = CsvReaderMock.return_value
    reader.len.return_value = 3  # valid indices: 0,1,2

    controller = TabularController(tmp_path)

    with pytest.raises(FedbiomedError) as exc:
        controller.get_sample(3)  # >= __len__()

    # Optional: sanity-check the error message contains the index
    assert "index 3" in str(exc.value)
