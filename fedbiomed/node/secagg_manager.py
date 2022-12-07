# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interfaces with the node secure aggregation element database table
"""
from abc import ABC
from typing import Union

from tinydb import TinyDB, Query

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.node.environ import environ


class SecaggManager(ABC):
    """Manage a node secagg element database table
    """
    def __init__(self):
        """Constructor of the class
        """
        self._db = TinyDB(environ['DB_PATH'])
        self._query = Query()
        self._table = None

    def get_by_secagg_id(self, secagg_id: str) -> Union[dict, None]:
        """Searches for data with given `secagg_id`

        There should be at most one entry with this unique secagg ID.

        Args:
            secagg_id: secure aggregation ID key to search

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`
        """
        # Trust argument check from `SecaggSetup``

        try:
            entries = self._table.search(
                self._query.secagg_id.exists() &
                (self._query.secagg_id == secagg_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB318.value}: failed searching the database table "{self._table}" ' \
                f'for secagg element with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        if (len(entries) > 1):
            errmess = f'{ErrorNumbers.FB318.value}: database table "{self._table}" is inconsistent: found {len(entries)} ' \
                f'entries with unique `secagg_id` {secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)            
        elif (len(entries) == 1):
            element = entries[0]
        else:
            element = None

        return element


class SecaggServkeyManager(SecaggManager):
    """Manage the node server key secagg element database table
    """
    def __init__(self):
        """Constructor of the class
        """
        super().__init__()

        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = self._db.table(name='SecaggServkey', cache_size=0)


class SecaggBiprimeManager(SecaggManager):
    """Manage the node biprime secagg element database table
    """
    def __init__(self):
        """Constructor of the class
        """
        super().__init__()

        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = self._db.table(name='SecaggBiprime', cache_size=0)
