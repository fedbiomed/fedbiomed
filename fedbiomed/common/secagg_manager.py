# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the component secure aggregation element database
"""
import os
from abc import ABC, abstractmethod
from typing import Union, List, Dict
import copy

import json
from tinydb import TinyDB, Query

from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.common.constants import ErrorNumbers, BiprimeType, SecaggElementTypes, SecureAggregationSchemes, __secagg_element_version__
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError, SchemeValidator


_TableName = 'SecaggManager'
_DefaultBiprimeValidator = SchemeValidator({
    'secagg_version': {"rules": [str], "required": True},
    'secagg_id': {"rules": [str], "required": True},
    'biprime': {"rules": [int], "required": True},
    'max_keysize': {"rules": [int], "required": True},
})


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
        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = self._db.table(name=_TableName, cache_size=0)

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
    def get(self, secagg_id: str, experiment_id: Union[str, None]):
        """Search for a data entry in component secagg element database"""

    def _add_generic(self,
                     secagg_scheme: SecureAggregationSchemes,
                     secagg_elem: SecaggElementTypes,
                     secagg_id: str,
                     parties: List[str],
                     specific: dict):
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
        self._check_existing_entry_in_db(secagg_id)
        specific.update({
            'secagg_version': str(__secagg_element_version__),
            'secagg_id': secagg_id,
            'parties': parties,
            'secagg_scheme': secagg_scheme.value,
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
        if self._get_generic(secagg_id) is not None:
            errmess = f'{ErrorNumbers.FB623.value}: error adding element in table "{self._table}": ' \
                      f' an entry already exists for secagg_id={secagg_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _raise_error_incompatible_retrieved_entry(self,
                                                  entry: Union[None, Dict],
                                                  experiment_id: str,
                                                  secagg_id: str,
                                                  component: SecaggElementTypes):
        if entry is not None and entry['experiment_id'] != experiment_id:
            errmess = f'{ErrorNumbers.FB623.value}: error getting {component.value} element: ' \
                      f'an entry exists for secagg_id={secagg_id} but does not belong to ' \
                      f'current experiment experiment_id={experiment_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _raise_error_mismatch_secagg_id_and_experiment_id(self,
                                                          entry: Union[None, Dict],
                                                          experiment_id: str,
                                                          secagg_id: str,
                                                          component: SecaggElementTypes):

        if entry is not None and entry['experiment_id'] != experiment_id:
            errmess = f'{ErrorNumbers.FB623.value}: error removing {component.value} element: ' \
                      f'an entry exists for secagg_id={secagg_id} but does not belong to ' \
                      f'current experiment experiment_id={experiment_id}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    @abstractmethod
    def add(self, secagg_id: str, parties: List[str], context: Dict[str, int], experiment_id: Union[str, None]):
        """Add a new data entry in component secagg element database"""

    def _remove_generic(self, secagg_id: str, component: SecaggElementTypes) -> bool:
        """Remove data entry for this `secagg_id` from database

        Args:
            secagg_id: secure aggregation ID key of the entry

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
    def remove(self, secagg_id: str, experiment_id: Union[str, None]) -> bool:
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

        Raises:
            FedbiomedSecaggError: the entry is associated with another experiment
        """

        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_retrieved_entry(element,
                                                       experiment_id,
                                                       secagg_id,
                                                       SecaggElementTypes.SERVER_KEY)

        return element

    def add(self, secagg_id: str, parties: List[str], context: Dict[str, int], experiment_id: str):
        """Add a new data entry for a context element in the servkey table 

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            experiment_id: ID of the experiment to which this secagg context element is attached
            context: server key part held by this party
        """

        # Trust argument type and value check from calling class (`SecaggSetup`, but not `Node`)
        self._add_generic(
            SecureAggregationSchemes.JOYE_LIBERT,
            SecaggElementTypes.SERVER_KEY,
            secagg_id,
            parties,
            {'experiment_id': experiment_id, 'context': context}
        )

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the server key table

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
        self._raise_error_mismatch_secagg_id_and_experiment_id(element,
                                                               experiment_id,
                                                               secagg_id,
                                                               SecaggElementTypes.SERVER_KEY)
        return self._remove_generic(secagg_id, SecaggElementTypes.SERVER_KEY)


class SecaggBiprimeManager(BaseSecaggManager):
    """Manage the component biprime secagg element database table
    """

    def __init__(self, db_path: str):
        """Constructor of the class

        Args:
            db_path: path to the component's secagg database
        """
        super().__init__(db_path)

        self._v = Validator()

    def is_default_biprime(self, secagg_id: str) -> bool:
        """Search for default (non dynamic) data entry with given `secagg_id` in the biprime table.

        Args:
            secagg_id: secure aggregation ID key to search

        Returns:
            True if a default biprime entry exists for this `secagg_id`, False if not
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        return isinstance(element, dict) and 'type' in element and element['type'] == BiprimeType.DEFAULT.value


    def get(self, secagg_id: str, experiment_id: None = None) -> Union[dict, None]:
        """Search for data entry with given `secagg_id` in the biprime table

        Check that there is at most one entry with this unique secagg ID.

        Args:
            secagg_id: secure aggregation ID key to search
            experiment_id: unused argument.
        Returns:
            A dict containing all values for the secagg element for this `secagg_id` if it exists,
                or None if no element exists for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        # type is internal to this class, need not transmit to caller
        if isinstance(element, dict) and 'type' in element:
            # `deepcopy` avoids  any risk of error related to database implementation
            element = copy.deepcopy(element)
            del element['type']

        return element

    def add(
            self,
            secagg_id: str,
            parties: List[str],
            context: Dict[str, int],
            experiment_id: None = None
    ) -> None:
        """Add a new data entry for a context element in the biprime table 

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            context: the (full) biprime number shared with other parties
            experiment_id: unused argument
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        self._add_generic(
            SecureAggregationSchemes.JOYE_LIBERT,
            SecaggElementTypes.BIPRIME,
            secagg_id,
            parties,
            {
                'context': context,
                'type': BiprimeType.DYNAMIC.value
            }
        )

    def remove(self, secagg_id: str, experiment_id: None = None) -> bool:
        """Remove data entry for this `secagg_id` from the biprime table

        Args:
            secagg_id: secure aggregation ID key of the entry
            experiment_id: unused argument
        Returns:
            True if an entry existed (and was removed) for this `secagg_id`,
                False if no entry existed for this `secagg_id`
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)

        # Can only remove dynamic biprimes
        element = self._get_generic(secagg_id)
        if isinstance(element, dict) and ('type' not in element or element['type'] != BiprimeType.DYNAMIC.value):
            errmess = f'{ErrorNumbers.FB623.value}: not authorized to remove non-dynamic biprime "{secagg_id}"'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        return self._remove_generic(secagg_id, SecaggElementTypes.BIPRIME)

    def _read_default_biprimes(self, default_biprimes_dir: str) -> List[Dict]:
        """Read default biprime files and check default biprime format

        Args:
            default_biprimes_dir: directory containing the default biprimes files

        Returns:
            a list of dictionaries, each one containing a default biprime

        Raises:
            FedbiomedSecaggError: cannot read biprime files
            FedbiomedSecaggError: badly formatted default biprime
        """
        default_biprimes = []

        for bp_file in os.listdir(default_biprimes_dir):
            if not bp_file.endswith('.json'):
                continue

            # Read default biprimes files
            logger.debug(f'Reading default biprime file "{bp_file}"')
            try:
                with open(os.path.join(default_biprimes_dir, bp_file)) as json_file:
                    biprime = json.load(json_file)
            except Exception as e:
                errmess = f'{ErrorNumbers.FB623.value}: cannot parse default biprime file "{bp_file}" : {e}'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

            # Check default biprimes content
            try:
                _DefaultBiprimeValidator.validate(biprime)
            except ValidatorError as e:
                errmess = f'{ErrorNumbers.FB623.value}: bad biprime format in file "{bp_file}": {e}'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

            if not biprime['secagg_id']:
                errmess = f'{ErrorNumbers.FB623.value}: bad biprime `secagg_id`` in file "{bp_file}" ' \
                          'must be a non-empty string'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

            default_biprimes.append(biprime)

        return default_biprimes

    def update_default_biprimes(self, allow_default_biprimes: bool, default_biprimes_dir: str) -> None:
        """Update the default entries in the biprime table.

        If `allow_default_biprimes` is True, then add or update the default biprimes from the *.json
            files in `default_biprimes_dir` directory.

        In all cases, remove the other default biprimes existing in the biprime table.

        Args:
            allow_default_biprimes: if True, then accept default biprimes from files
            default_biprimes_dir: directory containing the default biprimes files

        Raises:
            FedbiomedSecaggError: cannot update default biprimes
        """
        # Read and check the new proposed default biprime values from files
        if allow_default_biprimes:
            default_biprimes_new = self._read_default_biprimes(default_biprimes_dir)
        else:
            default_biprimes_new = []

        # Read the existing default biprimes in DB
        try:
            default_biprimes_current = self._table.search(
                self._query.type.exists() &
                (self._query.type == BiprimeType.DEFAULT.value)
            )
        except Exception as e:
            errmess = f'{ErrorNumbers.FB623.value}: database search operation failed for default biprimes: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # Remove existing default biprimes not in the new proposed values
        bp_new_ids = set(bp['secagg_id'] for bp in default_biprimes_new)
        bp_current_ids = set(bp['secagg_id'] for bp in default_biprimes_current)
        bp_remove_ids = list(bp_current_ids - bp_new_ids)

        try:
            self._table.remove(self._query.secagg_id.one_of(bp_remove_ids))
        except Exception as e:
            errmess = f'{ErrorNumbers.FB623.value}: database remove operation failed for ' \
                      f'obsolete default biprimes {bp_remove_ids}: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # Save or update the new default biprimes
        for bp in default_biprimes_new:
            try:
                self._table.upsert(
                    {
                        'secagg_version': bp['secagg_version'],
                        'secagg_id': bp['secagg_id'],
                        'parties': None,
                        'secagg_elem': SecaggElementTypes.BIPRIME.value,
                        'secagg_scheme': SecureAggregationSchemes.JOYE_LIBERT.value,
                        'type': BiprimeType.DEFAULT.value,
                        'context': {
                            'biprime': bp['biprime'],
                            'max_keysize': bp['max_keysize']
                        },
                    },
                    self._query.secagg_id == bp['secagg_id']
                )
            except Exception as e:
                errmess = f'{ErrorNumbers.FB623.value}: database upsert operation failed for ' \
                          f'default biprime {bp["secagg_id"]}: {e}'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)


class SecaggDhManager(BaseSecaggManager):
    """Manage the component Diffie Hellman secagg element database table
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

        Raises:
            FedbiomedSecaggError: the entry is associated with another experiment
        """
        # Trust argument type and value check from calling class (`SecaggSetup`, `Node`)
        element = self._get_generic(secagg_id)
        self._raise_error_incompatible_retrieved_entry(element, 
                                                       experiment_id,
                                                       secagg_id,
                                                       SecaggElementTypes.DIFFIE_HELLMAN)

        return element

    def add(self, secagg_id: str, parties: List[str], context: Dict[str, int], experiment_id: str):
        """Add a new data entry for a context element in the servkey table 

        Check that no entry exists yet for this `secagg_id` in the table.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            experiment_id: ID of the experiment to which this secagg context element is attached
            context: server key part held by this party
        """
        self._add_generic(
            SecureAggregationSchemes.LOM,
            SecaggElementTypes.DIFFIE_HELLMAN,
            secagg_id,
            parties,
            {'experiment_id': experiment_id, 'context': context}
        )

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Remove data entry for this `secagg_id` from the server key table

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
        self._raise_error_mismatch_secagg_id_and_experiment_id(element,
                                                               experiment_id,
                                                               secagg_id,
                                                               SecaggElementTypes.DIFFIE_HELLMAN)
        return self._remove_generic(secagg_id, SecaggElementTypes.DIFFIE_HELLMAN)
