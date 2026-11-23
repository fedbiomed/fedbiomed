import pytest
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import FedbiomedVersionError
from fedbiomed.node.channel_manager import ChannelManager
from fedbiomed.node.db_models import Channel


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.json")


def test_channel_manager_list_add_get(db_path):
    cm = ChannelManager(db_path)
    assert cm.list() == []
    assert cm.get("node1") is None

    cm.add("node1", b"12345678")
    cm.add("node1", b"replaced")

    assert cm.list() == ["node1"]
    element = cm.get("node1")
    assert element["distant_node_id"] == "node1"
    assert element["local_key"] == b"replaced"


def test_channel_manager_refuses_incompatible_version(db_path):
    cm = ChannelManager(db_path)
    with get_database(db_path).transaction() as session:
        session.add(
            Channel(distant_node_id="node1", n2n_channel_version="99.0", local_key="")
        )

    with pytest.raises(FedbiomedVersionError):
        cm.get("node1")
