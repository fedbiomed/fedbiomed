"""Secure Aggregation management on the researcher"""
import uuid
from typing import List, Union, Tuple
import time

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests

class SecaggContext:
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
        self._requests = Requests()
        self._status = False
        self._context = None

    def secagg_id(self) -> str:
        """Getter for secagg context element ID 

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    def status(self) -> bool:
        """Getter for secagg context element status

        Returns:
            `True` if secagg context element exists, `False` otherwise
        """
        return self._status

    # alternative: define method in subclass to have specific return type
    def context(self) -> Union[dict, None]:
        """Getter for secagg context element content

        Returns:
            secagg context element, or `None` if it doesn't exist
        """
        return self._context

    # TODO: subclass for specific
    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for secagg context element

        Returns:
            a tuple of a `context` and a `status` for the context element
        """
        logger.info('PUT RESEARCHER SECAGG SERVER_KEY PAYLOAD HERE')
        time.sleep(3)
        context = { 'msg': 'Not implemented yet' }
        status = True

        return context, status


    def setup(self) -> bool:
        """Setup secagg context element on defined parties.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.
        """
        # reset values in case `setup()` was already run and fails during this new execution
        self._status = False
        self._context = None

        msg = {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'parties': self._parties,
            'command': 'secagg',
        }
        for node in self._parties[1:]:
            self._requests.send_message(msg, node)
        status = {}

        # basic implementation: synchronous payload on researcher, then read answers from other parties
        context, status[self._researcher_id] = self._payload()

        # TODO: subclass to have specific payload for type

        # TODO: manage timeout
        # TODO: read answers
        responses = self._requests.get_responses(
            look_for_commands=['secagg'],
            # timeout=xxx
            only_successful=False
        )

        for resp in responses.data():
            # TODO check message fields
            if resp['node_id'] not in self._parties:
                errmess = f'{ErrorNumbers.FB414.value}: received message from node "{resp["node_id"]}"' \
                    'which is not a party of secagg "{self._secagg_id}"'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

            # this answer belongs to current secagg context setup
            status[resp['node_id']] = resp['success']

        if not set(status.keys()) == set(self._parties):
            # case where some parties did not answer
            # self._status = False
            # self._context = None
            absent = list(set(self._parties) - set(status.keys()))
            errmess = f'{ErrorNumbers.FB414.value}: some parties did not answer before timeout {absent}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        else:
            self._status = all(status.values())
            if self._status:
                self._context = context
            # else:
            #    self._context = None

        return self._status
