import pytest

from testsupport.fake_nipoppy_reader import FakeNipoppyReader


@pytest.fixture
def fake_nipoppy_reader(mocker):
    """Patches `NipoppyReader` in the controller module. Returns the created readers."""
    readers = []

    def _factory(path):
        reader = FakeNipoppyReader(path)
        readers.append(reader)
        return reader

    mocker.patch(
        "fedbiomed.common.dataset_controller._nipoppy_controller.NipoppyReader",
        side_effect=_factory,
    )
    return readers
