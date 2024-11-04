# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the component secure aggregation element database
"""
from abc import ABC, abstractmethod
from typing import Union, List, Dict
import base64

from tinydb import TinyDB, Query

from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.common.constants import (
    ErrorNumbers,
    SecaggElementTypes,
    __secagg_element_version__
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.singleton import SingletonMeta


_TableName = 'SecaggManager'


class _SecaggTableSingleton(metaclass=SingletonMeta):
    """Imstantiate secagg table object as singleton to ensure coherent acccess.
    """
    def __init__(self, db: TinyDB):
        """Constructor of the class

        Args:
            db: tinyDB database to use
        """
        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = db.table(name=_TableName, cache_size=0)

    @property
    def table(self) -> DBTable:
        """Getter for table"""
        return self._table


class BaseSecaggManager(ABC):
    """Manage a component secagg element database
    """

    def __init__(self, db_path: str):
        """Constructor of the class

        Args:
            db_path: path to the component's secagg database

        Raises:
            FedbiomedSecaggError: failed to access the database
        """
        try:
            self._db = TinyDB(db_path)
            self._db.table_class = DBTable
        except Exception as e:
            errmess = f'{ErrorNumbers.FB623.value}: failed to access the database with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._query = Query()
        self._table = _SecaggTableSingleton(self._db).table

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
            errmess = f'{ErrorNumbers.FB623.value}: failed searching the database table "{self._table}" ' \
                      f'for secagg element "{secagg_id}" with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        if len(entries) > 1:
            errmess = f'{ErrorNumbers.FB623.value}: database table "{self._table}" is inconsistent: ' \
                      f'found {len(entries)} entries with unique secagg_id={secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        elif len(entries) == 1:
            element = entries[0]
            raise_for_version_compatibility(
                element.get('secagg_version', __default_version__),
                __secagg_element_version__,
                f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
                f"secagg element {secagg_id} in database {self._table}"
            )
        else:
            element = None

        return element

    @abstractmethod
    def get(self, secagg_id: str, experiment_id: str):
        """Search for a data entry in component secagg element database"""

    def _add_generic(self,
                     secagg_elem: SecaggElementTypes,
                     secagg_id: str,
                     parties: List[str],
                     specific: dict):
        """Add a new data entry for this `secagg_id` in database

        Check that no entry exists yet for `secagg_id` in the table.

        Args:
            secagg_elem: type of secure aggregation component
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            specific: secagg data entry fields specific to this entry type

        Raises:
            FedbiomedSecaggError: failed to insert in database
        """
        self._check_existing_entry_in_db(secagg_id)
        specific.update({
            'secagg_version': str(__secagg_element_version__),
            'secagg_id': secagg_id,
            'parties': parties,
            'secagg_elem': secagg_elem.value
        })

        try:
            self._table.insert(specific)
        except Exception as e:
            errmess = f'{ErrorNumbers.FB623.value}: failed adding an entry in table "{self._table}" ' \
                      f'for secagg element={secagg_elem.name} secagg_id={secagg_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _check_existing_entry_in_db(self, secagg_id: str):
        """Checks if an entry with the secagg_id has already been saved in the database: if so,
        raises an error.

        Args:
            secagg_id: unique id used to save a given secure aggregation component

        Raises:
            FedbiomedSecaggError: if entry already exists
        """
        if self._get_generic(secagg_id) is not None:
            errmess = f'{ErrorNumbers.FB623.value}: error adding element in table "{self._table}": ' \
                      f' an entry already exists for secagg_id={secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _raise_error_incompatible_requested_entry(self,
                                                  entry: Union[None, Dict],
                                                  component: SecaggElementTypes,
                                                  secagg_id: str,
                                                  experiment_id: str,
                                                  database_operation_name: str = ''):
        """Raises error if:
            - there is a mismatch between the saved and the current Component
            - there is a mismatch between the saved and the current `experiment_id`

        Args:
            entry: entry of the database
            component: type of the element
            secagg_id: unique id used to save a given secure aggregation component
            experiment_id: id of the experiment.
            database_operation_name: string describing the operation taking place on the database.
                can be "getting", "removing"

        Raises:
            FedbiomedSecaggError: error raised if above condition(s) is/are matched.
        """
        errmess: str = None
        if entry is not None:
            if entry['experiment_id'] != experiment_id:
                errmess = f'{ErrorNumbers.FB623.value}: error {database_operation_name} {component.name} element: ' \
                          f'an entry exists for secagg_id={secagg_id} but does not belong to ' \
                          f'current experiment experiment_id={experiment_id}'

            if entry['secagg_elem'] != component.value:
                errmess = f'{ErrorNumbers.FB623.value}: error {database_operation_name} {component.name} element: ' \
                          f'an entry exists for secagg_id={secagg_id} and  experiment_id={experiment_id}' \
                          f' but was saved as a {SecaggElementTypes.get_element_from_value(entry["secagg_elem"])}'

            if errmess:
                raise FedbiomedSecaggError(errmess)

    @abstractmethod
    def add(self, secagg_id: str, parties: List[str], context: Dict[str, int], experiment_id: str):
        """Add a new data entry in component secagg element database"""

    def _remove_generic(self, secagg_id: str, component: SecaggElementTypes) -> bool:
        """Remove data entry for this `secagg_id` from database

        Args:
            secagg_id: secure aggregation ID key of the entry
            component: type of the element

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`

        Raises:
            FedbiomedSecaggError: failed to remove entry from the database
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
            errmess = f'{ErrorNumbers.FB623.value}: failed removing an entry from table "{self._table}" ' \
                      f'for secagg element {component.value} secagg_id={secagg_id} with error: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        return True

    @abstractmethod
    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Remove a data entry from component secagg element database"""


class SecaggServkeyManager(BaseSecaggManager):
    """Manage the component server key secagg element database table
    """

    def __init__(self, db_path: str):
        """Constructor of the class

        Args:
            db_path: path to the component's secagg database
        """
        super().__init__(db_path)

    def get(self, secagg_id: str, experiment_id: str) -> Union[dict, None]:
        """Search for data entry with given `secagg_id`

        Check that there is at most one entry with this unique secagg ID.

        If there is an entry for this `secagg_id`, check it is associated with experiment `experiment_id`

        Args:
            secagg_id: secure aggregation ID key to search
            experiment_id: the experiment ID associated with the secagg entry

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`
        """

        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_requested_entry(element,
                                                       SecaggElementTypes.SERVER_KEY,
                                                       secagg_id,
                                                       experiment_id,
                                                       'getting')

        return element

    def add(self, secagg_id: str, parties: List[str], context: Dict[str, int], experiment_id: str):
        """Add a new data entry for a context element in the secagg table

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            context: server key part held by this party
            experiment_id: ID of the experiment to which this secagg context element is attached
        """

        # Trust argument type and value check from calling class (`SecaggSetup`, but not `Node`)
        self._add_generic(
            SecaggElementTypes.SERVER_KEY,
            secagg_id,
            parties,
            {'experiment_id': experiment_id, 'context': context}
        )

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the secagg table

        Check that the experiment ID for the table entry and the current experiment match

        Args:
            secagg_id: secure aggregation ID key of the entry
            experiment_id: experiment ID of the current experiment

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`
        """

        # Trust argument type and value check from calling class for `secagg_id` (`SecaggSetup`, but not `Node`)
        # Don't trust `Node` for `experiment_id` type (may give `None`) but this is not an issue
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_requested_entry(element,
                                                       SecaggElementTypes.SERVER_KEY,
                                                       secagg_id,
                                                       experiment_id,
                                                       'removing')
        return self._remove_generic(secagg_id, SecaggElementTypes.SERVER_KEY)


class SecaggDhManager(BaseSecaggManager):
    # FIXME: this should be called `SecaggDHManager`
    """Manage the secagg table elements for Diffie Hellman components
    """

    def get(self, secagg_id: str, experiment_id: str) -> Union[dict, None]:
        """Search for data entry with given `secagg_id`

        Check that there is at most one entry with this unique secagg ID.

        If there is an entry for this `secagg_id`, check it is associated with experiment `experiment_id`

        Args:
            secagg_id: secure aggregation ID key to search
            experiment_id: the experiment ID associated with the secagg entry

        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_requested_entry(element,
                                                       SecaggElementTypes.DIFFIE_HELLMAN,
                                                       secagg_id,
                                                       experiment_id,
                                                       'getting')

        if element:
            # Need to convert to keys as bytes
            context_bytes = {
                node_id: bytes(base64.b64decode(key)) \
                    for node_id, key in element['context'].items()}
            element['context'] = context_bytes

        return element

    def add(self, secagg_id: str, parties: List[str], context: Dict[str, bytes], experiment_id: str):
        """Add a new data entry for a context element in the secagg table

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            experiment_id: ID of the experiment to which this secagg context element is attached
            context: server key part held by this party
        """
        # Save key pairs as `str`` since it is the format support by JSON. Need to convert to `base64` first
        context_json = {node_id: str(base64.b64encode(key), 'utf-8') for node_id, key in context.items()}

        self._add_generic(
            SecaggElementTypes.DIFFIE_HELLMAN,
            secagg_id,
            parties,
            {'experiment_id': experiment_id, 'context': context_json}
        )

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the secagg table

        Check that the experiment ID for the table entry and the current experiment match

        Args:
            secagg_id: secure aggregation ID key of the entry
            experiment_id: experiment ID of the current experiment

        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`

        Raises:
            FedbiomedSecaggError: database entry does not belong to `experiment_id`
        """
        # Trust argument type and value check from calling class for `secagg_id` (`SecaggSetup`, but not `Node`)
        # Don't trust `Node` for `experiment_id` type (may give `None`) but this is not an issue
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_requested_entry(element,
                                                       SecaggElementTypes.DIFFIE_HELLMAN,
                                                       secagg_id,
                                                       experiment_id,
                                                       'removing')
        return self._remove_generic(secagg_id, SecaggElementTypes.DIFFIE_HELLMAN)
