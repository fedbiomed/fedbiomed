# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the node to node channel state database"""

from typing import List, Union

from sqlalchemy import select

from fedbiomed.common.constants import ErrorNumbers, __n2n_channel_element_version__
from fedbiomed.common.db import get_database
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import raise_for_version_compatibility
from fedbiomed.node.db_entries import ChannelEntry


class ChannelManager:
    """Keeps the local keys of the node's channels to other nodes.

    Each public method runs in one transaction of the node's database.
    """

    # Tables this manager owns; the only ones it reads or writes
    MODELS = (ChannelEntry,)

    def __init__(self, db_path: str):
        """Initialize with the node's database, creating the missing `MODELS` tables.

        Args:
            db_path: Path of the node's database file, as in its config.
        """
        self._database = get_database(db_path)
        self._database.create_tables(self.MODELS)

    def list(self) -> List[str]:
        """Returns the `distant_node_id` of all known channels."""
        with self._database.transaction() as session:
            channels = list(session.scalars(select(ChannelEntry.distant_node_id)))

        logger.debug(f"Listed node-to-node channels: count={len(channels)}")
        return channels

    def get(self, distant_node_id: str) -> Union[dict, None]:
        """Returns the channel to `distant_node_id`, or None if there is none.

        Raises:
            FedbiomedError: the channel was saved by an incompatible version
        """
        with self._database.transaction() as session:
            channel = session.get(ChannelEntry, distant_node_id)
            element = channel.to_dict() if channel is not None else None

        logger.debug(
            f"Lookup node-to-node channel: distant_node_id={distant_node_id} found={element is not None}"
        )

        if element:
            raise_for_version_compatibility(
                element["n2n_channel_version"],
                __n2n_channel_element_version__,
                f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
                f"channel distant_node_id={distant_node_id} in table {ChannelEntry.__tablename__}",
            )

        return element

    def add(self, distant_node_id: str, local_key: bytes):
        """Saves the channel to `distant_node_id`, replacing the existing one if any.

        Args:
            distant_node_id: unique id used to save a given channel component
            local_key: private key of local endpoint for this channel
        """
        with self._database.transaction() as session:
            session.merge(
                ChannelEntry(
                    distant_node_id=distant_node_id,
                    local_key=local_key,
                    n2n_channel_version=str(__n2n_channel_element_version__),
                )
            )

        logger.debug(
            f"Saved node-to-node channel metadata: distant_node_id={distant_node_id}"
        )
