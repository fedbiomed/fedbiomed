import pytest
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.channel_manager import ChannelManager
from fedbiomed.node.db_entries import ChannelEntry


@pytest.fixture
def manager(tmp_path):
    return ChannelManager(str(tmp_path / "node.json"))


def test_no_channel_is_known_at_start(manager):
    assert manager.list() == []
    assert manager.get("node_2") is None


def test_added_channel_is_listed_and_returned(manager):
    manager.add("node_2", b"\x00key")

    assert manager.list() == ["node_2"]
    channel = manager.get("node_2")
    assert channel["distant_node_id"] == "node_2"
    assert channel["local_key"] == b"\x00key"


def test_adding_a_known_channel_replaces_its_key(manager):
    manager.add("node_2", b"old")
    manager.add("node_2", b"new")

    assert manager.list() == ["node_2"]
    assert manager.get("node_2")["local_key"] == b"new"


def test_channels_persist_across_managers(tmp_path):
    ChannelManager(str(tmp_path / "node.json")).add("node_2", b"key")

    assert ChannelManager(str(tmp_path / "node.json")).list() == ["node_2"]


def test_channel_of_incompatible_version_is_refused(manager, tmp_path):
    with get_database(str(tmp_path / "node.json")).transaction() as session:
        session.add(
            ChannelEntry(
                distant_node_id="node_2", local_key=b"key", n2n_channel_version="99"
            )
        )

    with pytest.raises(FedbiomedError, match="FB625"):
        manager.get("node_2")
