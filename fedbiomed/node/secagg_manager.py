# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the node secure aggregation element database
"""
from abc import ABC, abstractmethod
from typing import Union, List

from tinydb import TinyDB, Query

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.node.environ import environ


class SecaggManager(ABC):
    """Manage a node secagg element database
    """
    def __init__(self):
        """Constructor of the class

        Raises:
            FedbiomedSecaggError: failed to access the database
        """
        try:
            self._db = TinyDB(environ['DB_PATH'])
        except Exception as e:
            errmess = f'{ErrorNumbers.FB318.value}: failed to access the database with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess) 

        self._query = Query()
        self._table = None

    def _get_generic(self, secagg_id: str) -> Union[dict, None]:
        """Search for data entry with given `secagg_id`

        Check that there is at most one entry with this unique secagg ID.

        Args:
            secagg_id: secure aggregation ID key to search

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`

        Raises:
            FedbiomedSecaggError: failed to query the database
            FedbiomedSecaggError: more than one entry in database with this secagg ID
        """
        try:
            entries = self._table.search(
                self._query.secagg_id.exists() &
                (self._query.secagg_id == secagg_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB318.value}: failed searching the database table "{self._table}" ' \
                f'for secagg element "{secagg_id}" with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        if (len(entries) > 1):
            errmess = f'{ErrorNumbers.FB318.value}: database table "{self._table}" is inconsistent: ' \
                f'found {len(entries)} entries with unique secagg_id={secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)            
        elif (len(entries) == 1):
            element = entries[0]
        else:
            element = None

        return element

    @abstractmethod
    def get(self, secagg_id: str, *args):
        """Search for a data entry in node secagg element database"""

    def _add_generic(self, secagg_id: str, parties: List[str], specific: dict):
        """Add a new data entry for this `secagg_id` in database

        Check that no entry exists yet for `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            specific: secagg data entry fields specific to this entry type 

        Raises:
            FedbiomedSecaggError: failed to insert in database
            FedbiomedSecaggError: an entry already exists for `secagg_id` in the table
        """
        if self._get_generic(secagg_id) is not None:
            errmess = f'{ErrorNumbers.FB318.value}: error adding element in table "{self._table}": ' \
                f' an entry already exists for secagg_id={secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)              

        specific.update({'secagg_id': secagg_id, 'parties': parties})
        try:
            self._table.insert(specific)
        except Exception as e:
            errmess = f'{ErrorNumbers.FB318.value}: failed adding an entry in table "{self._table}" ' \
                f'for secagg element secagg_id={secagg_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)    

    @abstractmethod
    def add(self, secagg_id: str, parties: List[str], *args):
        """Add a new data entry in node secagg element database"""

    def _remove_generic(self, secagg_id: str) -> bool:
        """Remove data entry for this `secagg_id` from database

        Args:
            secagg_id: secure aggregation ID key of the entry

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`
        """
        # Rely on element found in database (rather than number of element removed)
        if self._get_generic(secagg_id) is None:
            return False

        try:
            # we could also test number of elements deleted for double check
            self._table.remove(
                self._query.secagg_id.exists() &
                (self._query.secagg_id == secagg_id)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB318.value}: failed removing an entry from table "{self._table}" ' \
                f'for secagg element secagg_id={secagg_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)    

        return True

    @abstractmethod
    def remove(self, secagg_id: str, *args) -> bool:
        """Remove a data entry from node secagg element database"""


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

    def get(self, secagg_id: str, job_id: str) -> Union[dict, None]:
        """Search for data entry with given `secagg_id`

        Check that there is at most one entry with this unique secagg ID.

        If there is an entry for this `secagg_id`, check it is associated with job `job_id`

        Args:
            secagg_id: secure aggregation ID key to search
            job_id: the job ID associated with the secagg entry

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`

        Raises:
            FedbiomedSecaggError: the entry is associated with another job
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        if element is not None and element['job_id'] != job_id:
            errmess = f'{ErrorNumbers.FB318.value}: error getting servkey element: ' \
                f'an entry exists for secagg_id={secagg_id} but does not belong to ' \
                f'current job job_id={job_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess) 

        return element

    def add(self, secagg_id: str, parties: List[str], job_id: str, servkey_share: str):
        """Add a new data entry for a context element in the servkey table 

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            job_id: ID of the job to which this secagg context element is attached
            servkey_share: server key part held by this party
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        self._add_generic(
            secagg_id,
            parties,
            {'job_id': job_id, 'servkey_share': servkey_share }
        )

    def remove(self, secagg_id: str, job_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the server key table

        Check that the job ID for the table entry and the current job match  

        Args:
            secagg_id: secure aggregation ID key of the entry
            job_id: job ID of the current job

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        if element is not None and element['job_id'] != job_id:
            errmess = f'{ErrorNumbers.FB318.value}: error removing servkey element: ' \
                f'an entry exists for secagg_id={secagg_id} but does not belong to ' \
                f'current job job_id={job_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess) 

        return self._remove_generic(secagg_id)


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

    def get(self, secagg_id: str) -> Union[dict, None]:
        """Search for data entry with given `secagg_id` in the biprime table

        Check that there is at most one entry with this unique secagg ID.

        Args:
            secagg_id: secure aggregation ID key to search

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        return self._get_generic(secagg_id)

    def add(self, secagg_id: str, parties: List[str], biprime: str):
        """Add a new data entry for a context element in the biprime table 

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            biprime: the (full) biprime number shared with other parties
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        self._add_generic(
            secagg_id,
            parties,
            {'biprime': biprime }
        )

    def remove(self, secagg_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the biprime table

        Args:
            secagg_id: secure aggregation ID key of the entry

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        return self._remove_generic(secagg_id)
