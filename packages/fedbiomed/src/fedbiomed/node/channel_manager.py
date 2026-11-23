# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the node to node channel state database"""

import base64
from typing import List, Union

from sqlalchemy import select

from fedbiomed.common.constants import ErrorNumbers, __n2n_channel_element_version__
from fedbiomed.common.db import get_database
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import raise_for_version_compatibility
from fedbiomed.node.db_models import Channel


class ChannelManager:
    """Keeps the local key of each node to node channel.

    Each public method runs in one transaction of the component's database.
    """

    # Tables this manager owns; the only ones it reads or writes
    MODELS = (Channel,)

    def __init__(self, db_path: str):
        """Initialize with the component's database, creating the missing `MODELS` tables.

        Args:
            db_path: Path of the component's database file, as in its config.
        """
        self._database = get_database(db_path)
        self._database.create_tables(self.MODELS)

    def list(self) -> List[str]:
        """List all channels in the database.

        Returns:
            List of `distant_node_id` of all known channels
        """
        with self._database.transaction() as session:
            channels = list(session.scalars(select(Channel.distant_node_id)))

        logger.debug(f"Listed node-to-node channels: count={len(channels)}")
        return channels

    def get(self, distant_node_id: str) -> Union[dict, None]:
        """Gets the channel with a `distant_node_id`.

        Args:
            distant_node_id: unique id used to save a given channel component

        Returns:
            A dict containing all values for the channel for this `distant_node_id` if it exists,
                or None if no channel exists for this `distant_node_id`

        Raises:
            FedbiomedVersionError: the channel was saved by an incompatible version
        """
        with self._database.transaction() as session:
            channel = session.get(Channel, distant_node_id)
            element = None if channel is None else channel.to_dict()

        logger.debug(
            f"Lookup node-to-node channel: distant_node_id={distant_node_id} found={element is not None}"
        )

        if element:
            raise_for_version_compatibility(
                element["n2n_channel_version"],
                __n2n_channel_element_version__,
                f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
                f"channel distant_node_id={distant_node_id}",
            )
            element["local_key"] = base64.b64decode(element["local_key"])

        return element

    def add(self, distant_node_id: str, local_key: bytes):
        """Saves the channel with a `distant_node_id`, replacing an existing one.

        Args:
            distant_node_id: unique id used to save a given channel component
            local_key: private key of local endpoint for this channel
        """
        channel = Channel(
            distant_node_id=distant_node_id,
            n2n_channel_version=str(__n2n_channel_element_version__),
            local_key=str(base64.b64encode(local_key), "utf-8"),
        )
        with self._database.transaction() as session:
            session.merge(channel)

        logger.debug(
            f"Saved node-to-node channel metadata: distant_node_id={distant_node_id}"
        )
