"""Secure Aggregation management on the researcher"""
import uuid
from typing import List, Any

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests

class SecaggContext(object):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """

    def __init__(self, element: SecaggElementTypes, parties: List[str]):
        """Constructor of the class.

        Args:
            element: kind of context element handled by this object
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher

        Raises:
            FedbiomedSecaggError: xxx
        """
        # TODO: check types and values

        self._secagg_id = 'secagg_' + str(uuid.uuid4())
        self._element = element
        self._parties = parties

        self._researcher_id = environ['RESEARCHER_ID']
        self._context = None
        self._requests = Requests()

    def secagg_id(self) -> str:
        """Getter for secagg context element ID 

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    # TODO: subclass to be able to type returned value
    def context(self) -> Any:
        """Getter for secagg context element

        Returns:
            secagg context element
        """

    def setup(self) -> bool:
        """Setup secagg context element on defined parties.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.
        """

        msg = {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'parties': self._parties,
            'command': 'secagg',
        }
        for node in self._parties[1:]:
            self._requests.send_message(msg, node)

        # basic implementation: synchronous payload on researcher, then read answers from other parties
        # TODO: payload on researcher
        # TODO: subclass to have specific payload for type

        # TODO: read answers

        return False
