# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interface with the component secure aggregation element database"""

import base64
from typing import Dict, List, Union

from fedbiomed.common.constants import (
    ErrorNumbers,
    SecaggElementTypes,
    __secagg_element_version__,
)
from fedbiomed.common.db import get_database
from fedbiomed.common.db_models import SecaggElementEntry
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.utils import raise_for_version_compatibility


class BaseSecaggManager:
    """Keeps the component's secure aggregation context elements of one type.

    Each public method runs in one transaction of the component's database.
    """

    # Tables this manager owns; the only ones it reads or writes
    MODELS = (SecaggElementEntry,)
    # Type of the elements this manager keeps
    _ELEMENT: SecaggElementTypes

    def __init__(self, db_path: str):
        """Initialize with the component's database, creating the missing `MODELS` tables.

        Args:
            db_path: Path of the component's database file, as in its config.
        """
        self._database = get_database(db_path)
        self._database.create_tables(self.MODELS)

    def get(self, secagg_id: str, experiment_id: str) -> Union[dict, None]:
        """Gets the element with a `secagg_id`.

        Args:
            secagg_id: secure aggregation ID key to search
            experiment_id: the experiment ID associated with the secagg entry

        Returns:
            The element, or None if no element exists for this `secagg_id`

        Raises:
            FedbiomedSecaggError: the element belongs to another experiment or type
        """
        with self._database.transaction() as session:
            element = session.get(SecaggElementEntry, secagg_id)
            element = None if element is None else element.to_dict()

        if element is not None:
            self._check(element, experiment_id, "getting")
        return element

    def add(
        self,
        secagg_id: str,
        parties: List[str],
        context: Dict[str, int],
        experiment_id: str,
    ):
        """Saves a new element; an element with the same `secagg_id` is refused.

        Args:
            secagg_id: secure aggregation ID key of the entry
            parties: list of parties participating in this secagg context element
            context: secret held by this party
            experiment_id: ID of the experiment to which this secagg context element is attached
        """
        with self._database.transaction() as session:
            session.add(
                SecaggElementEntry(
                    secagg_id=secagg_id,
                    secagg_version=str(__secagg_element_version__),
                    secagg_elem=self._ELEMENT.value,
                    experiment_id=experiment_id,
                    parties=parties,
                    context=context,
                )
            )

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        """Removes the element with a `secagg_id`.

        Args:
            secagg_id: secure aggregation ID key of the entry
            experiment_id: experiment ID of the current experiment

        Returns:
            True if an element existed (and was removed), False otherwise

        Raises:
            FedbiomedSecaggError: the element belongs to another experiment or type
        """
        with self._database.transaction() as session:
            element = session.get(SecaggElementEntry, secagg_id)
            if element is None:
                return False
            self._check(element.to_dict(), experiment_id, "removing")
            session.delete(element)
        return True

    def _check(self, element: dict, experiment_id: str, operation: str) -> None:
        """Refuses an element of another version, experiment or type."""
        secagg_id = element["secagg_id"]
        raise_for_version_compatibility(
            element["secagg_version"],
            __secagg_element_version__,
            f"{ErrorNumbers.FB625.value}: Incompatible versions  (found %s but expected %s) for "
            f"secagg element {secagg_id}",
        )
        if element["experiment_id"] != experiment_id:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB623.value}: error {operation} {self._ELEMENT.name} element: "
                f"an entry exists for secagg_id={secagg_id} but does not belong to "
                f"current experiment experiment_id={experiment_id}"
            )
        if element["secagg_elem"] != self._ELEMENT.value:
            saved = SecaggElementTypes.get_element_from_value(element["secagg_elem"])
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB623.value}: error {operation} {self._ELEMENT.name} element: "
                f"an entry exists for secagg_id={secagg_id} and  experiment_id={experiment_id}"
                f" but was saved as a {saved}"
            )


class SecaggServkeyManager(BaseSecaggManager):
    """Keeps the component's server key secagg elements."""

    _ELEMENT = SecaggElementTypes.SERVER_KEY


class SecaggDhManager(BaseSecaggManager):
    # FIXME: this should be called `SecaggDHManager`
    """Keeps the component's Diffie-Hellman secagg elements; keys stored in base64."""

    _ELEMENT = SecaggElementTypes.DIFFIE_HELLMAN

    def get(self, secagg_id: str, experiment_id: str) -> Union[dict, None]:
        """Gets the element with a `secagg_id`, its keys as bytes."""
        element = super().get(secagg_id, experiment_id)
        if element is not None:
            element["context"] = {
                node_id: base64.b64decode(key)
                for node_id, key in element["context"].items()
            }
        return element

    def add(
        self,
        secagg_id: str,
        parties: List[str],
        context: Dict[str, bytes],
        experiment_id: str,
    ):
        """Saves a new element, its keys in base64 as JSON holds no bytes."""
        context_json = {
            node_id: str(base64.b64encode(key), "utf-8")
            for node_id, key in context.items()
        }
        super().add(secagg_id, parties, context_json, experiment_id)
