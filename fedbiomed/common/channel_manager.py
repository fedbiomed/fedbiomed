# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the node to node channel state database
"""
from typing import Union, List
import base64

from tinydb import TinyDB, Query

from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.common.constants import ErrorNumbers, __n2n_channel_element_version__
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError


_TableName = 'ChannelManager'


class ChannelManager:
    """Manage a node to node channel status database
    """

    def __init__(self, db_path: str):
        """Constructor of the class

        Args:
            db_path: path to the n2n channel database

        Raises:
            FedbiomedNodeToNodeError: failed to access the database
        """
        try:
            self._db = TinyDB(db_path)
            self._db.table_class = DBTable
        except Exception as e:
            errmess = f'{ErrorNumbers.FB631.value}: failed to access the database with error: {e}'
            raise FedbiomedNodeToNodeError(errmess) from e

        self._query = Query()
        self._table = self._db.table(name=_TableName, cache_size=0)

    def list(self) -> List[str]:
        """List all channels in the database.

        Returns:
            List of `distant_node_id` of all known channels

        Raises:
            FedbiomedNodeToNodeError: failed to query the database
        """
        try:
            channels = [ channel['distant_node_id'] for channel in self._table.all() ]
        except Exception as e:
            errmess = f'{ErrorNumbers.FB631.value}: failed listing elements in the database table "{self._table}" ' \
                      f'with error: {e}'
            raise FedbiomedNodeToNodeError(errmess) from e

        return channels

    def get(self, distant_node_id: str) -> Union[dict, None]:
        """Search for data entry with given `distant_node_id`

        Don't check that there is at most one entry with this unique distant node ID.

        Args:
            distant_node_id: unique id used to save a given channel component

        Returns:
            A dict containing all values for the channel for this `distant_node_id` if it exists,
                or None if no channel exists for this `distant_node_id`

        Raises:
            FedbiomedNodeToNodeError: failed to query the database
        """
        try:
            element = self._table.get(
                self._query.distant_node_id.exists() &
                (self._query.distant_node_id == distant_node_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB631.value}: failed searching the database table "{self._table}" ' \
                      f'for channel distant_node_id="{distant_node_id}" with error: {e}'
            raise FedbiomedNodeToNodeError(errmess) from e

        if element:
            raise_for_version_compatibility(
                element.get('n2n_channel_version', __default_version__),
                __n2n_channel_element_version__,
                f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
                f"channel distant_node_id={distant_node_id} in database {self._table}"
            )

            # Need to convert to bytes
            element['local_key'] = bytes(base64.b64decode(element['local_key']))

        return element

    def add(self, distant_node_id: str, local_key: bytes):
        """Add a new data entry for this `distant_node_id` in database

        If an entry already exists, update it.

        Args:
            distant_node_id: unique id used to save a given channel component
            local_key: private key of local endpoint for this channel

        Raises:
            FedbiomedNodeToNodeError: failed to insert in database
        """
        channel = {
            'n2n_channel_version': str(__n2n_channel_element_version__),
            'distant_node_id': distant_node_id,
            'local_key': str(base64.b64encode(local_key), 'utf-8'),
        }

        try:
            self._table.upsert(channel, self._query.distant_node_id == distant_node_id)
        except Exception as e:
            errmess = f'{ErrorNumbers.FB631.value}: failed adding an entry in table "{self._table}" ' \
                      f'for distant_node_id={distant_node_id} with error: {e}'
            raise FedbiomedNodeToNodeError(errmess) from e
