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
from fedbiomed.common.logger import logger


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
            errmess = f'{ErrorNumbers.FB630.value}: failed to access the database with error: {e}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

        self._query = Query()
        self._table = self._db.table(name=_TableName, cache_size=0)


    def _check_existing_entry_in_db(self, distant_node_id: str):
        """Checks if an entry with the distant_node_id has already been saved in the database: if so,
        raises an error.

        Args:
            distant_node_id: unique id used to save a given channel component

        Raises:
            FedbiomedNodeToNodeError: if entry already exists
        """
        if self.get(distant_node_id) is not None:
            errmess = f'{ErrorNumbers.FB630.value}: error adding element in table "{self._table}": ' \
                      f' an entry already exists for distant_node_id={distant_node_id}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

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
            errmess = f'{ErrorNumbers.FB630.value}: failed listing elements in the database table "{self._table}" ' \
                      f'with error: {e}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

        return channels

    def get(self, distant_node_id: str) -> Union[dict, None]:
        """Search for data entry with given `distant_node_id`

        Check that there is at most one entry with this unique distant node ID.

        Args:
            distant_node_id: unique id used to save a given channel component

        Returns:
            A dict containing all values for the channel for this `distant_node_id` if it exists,
                or None if no channel exists for this `distant_node_id`

        Raises:
            FedbiomedNodeToNodeError: failed to query the database
            FedbiomedNodeToNodeError: more than one entry in database with this node ID
        """
        try:
            entries = self._table.search(
                self._query.distant_node_id.exists() &
                (self._query.distant_node_id == distant_node_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB630.value}: failed searching the database table "{self._table}" ' \
                      f'for channel distant_node_id="{distant_node_id}" with error: {e}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

        if len(entries) > 1:
            errmess = f'{ErrorNumbers.FB630.value}: database table "{self._table}" is inconsistent: ' \
                      f'found {len(entries)} entries with unique distant_node_id={distant_node_id}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)
        elif len(entries) == 1:
            element = entries[0]
            raise_for_version_compatibility(
                element.get('n2n_channel_version', __default_version__),
                __n2n_channel_element_version__,
                f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
                f"channel distant_node_id={distant_node_id} in database {self._table}"
            )
        else:
            element = None

        if element:
            # Need to convert to bytes
            element['local_key'] = bytes(base64.b64decode(element['local_key']))

        return element

    def add(self, distant_node_id: str, local_key: bytes):
        """Add a new data entry for this `distant_node_id` in database

        Check that no entry exists yet for `distant_node_id` in the table.

        Args:
            distant_node_id: unique id used to save a given channel component
            local_key: private key of local endpoint for this channel

        Raises:
            FedbiomedNodeToNodeError: failed to insert in database
        """
        self._check_existing_entry_in_db(distant_node_id)
        channel = {
            'n2n_channel_version': str(__n2n_channel_element_version__),
            'distant_node_id': distant_node_id,
            'local_key': str(base64.b64encode(local_key), 'utf-8'),
        }

        try:
            self._table.insert(channel)
        except Exception as e:
            errmess = f'{ErrorNumbers.FB630.value}: failed adding an entry in table "{self._table}" ' \
                      f'for distant_node_id={distant_node_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

    def remove(self, distant_node_id: str) -> bool:
        """Remove data entry for this `distant_node_id` from database

        Args:
            distant_node_id: unique id used to save a given channel component

        Returns:
            True if an entry existed (and was removed) for this `distant_node_id`,
                False if no entry existed for this `distant_node_id`

        Raises:
            FedbiomedNodeToNodeError: failed to remove entry from the database
        """
        # Rely on element found in database (rather than number of element removed)
        if self.get(distant_node_id) is None:
            return False

        try:
            # we could also test number of elements deleted for double check
            self._table.remove(
                self._query.distant_node_id.exists() &
                (self._query.distant_node_id == distant_node_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB630.value}: failed removing an entry from table "{self._table}" ' \
                      f'for channel distant_node_id={distant_node_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedNodeToNodeError(errmess)

        return True
