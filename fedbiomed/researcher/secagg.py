"""Secure Aggregation management on the researcher"""
import uuid
from typing import Callable, List, Union, Tuple
from abc import ABC, abstractmethod
import time

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests


class SecaggContext(ABC):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """

    def __init__(self, parties: List[str]):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher

        Raises:
            FedbiomedSecaggError: TODO
        """
        # TODO: check types and values

        self._secagg_id = 'secagg_' + str(uuid.uuid4())
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

    @abstractmethod
    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for secagg context element

        Returns:
            a tuple of a `context` and a `status` for the context element
        """
        pass


    def _secagg_round(self, msg: dict, command: str, payload: Callable, timeout: float = 0) -> bool:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round
            command: reply command expected from the parties
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action
            timeout: maximum duration for the negotiation phase. Defaults to `environ['TIMEOUT']` if unser
                or equals 0.

        Raises:
            FedbiomedSecaggError: some parties did not answer before timeout
            FedbiomedSecaggError: received a reply for a non-party to the negotiation

        Returns:
            True if secagg context element action could be done for all parties, False if at least
                one of the parties could not do the context element action.
        """
        # reset values in case `setup()` was already run (and fails during this new execution,
        # or this is a deletion)
        self._status = False
        self._context = None
        timeout = timeout or environ['TIMEOUT']
        start_time = time.time()

        sequence = {}
        for node in self._parties[1:]:
            sequence[node] = self._requests.send_message(msg, node, add_sequence=True)
        status = {}

        # basic implementation: synchronous payload on researcher, then read answers from other parties
        context, status[self._researcher_id] = payload()

        while True:
            # wait at most until `timeout` by chunks <= 1 second
            remain_time = start_time + timeout - time.time()
            if remain_time <= 0:
                break
            wait_time = min(1, remain_time)
            responses = self._requests.get_responses(
                look_for_commands=[command],
                timeout=wait_time,
                only_successful=False,
                while_responses=False
            )

            for resp in responses.data():
                # order of test matters !
                if resp['researcher_id'] != self._researcher_id:
                    break
                if resp['secagg_id'] != self._secagg_id:
                    logger.debug(
                        f"Unexpected secagg reply: expected `secagg_id` {self._secagg_id}"
                        f" and received {resp['secagg_id']}")
                    break
                if resp['node_id'] not in self._parties[1:]:
                    errmess = f'{ErrorNumbers.FB414.value}: received message from node "{resp["node_id"]}"' \
                        'which is not a party of secagg "{self._secagg_id}"'
                    logger.error(errmess)
                    raise FedbiomedSecaggError(errmess)
                if resp['sequence'] != sequence[resp['node_id']]:
                    logger.debug(
                        f"Out of sequence secagg reply: expected `sequence` {sequence[resp['node_id']]}"
                        f" and received {resp['sequence']}"
                    )

                # this answer belongs to current secagg context setup
                status[resp['node_id']] = resp['success']

            if set(status.keys()) == set(self._parties):
                break

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

    def setup(self, timeout: float = 0) -> bool:
        """Setup secagg context element on defined parties.

        Args:
            timeout: maximum duration for the setup phase. Defaults to `environ['TIMEOUT']` if unset
                or equals 0.

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
        return self._secagg_round(msg, 'secagg', self._payload, timeout)


class SecaggServkeyContext(SecaggContext):
    """
    Handles a Secure Aggregation server key context element on the researcher side.
    """

    def __init__(self, parties: List[str]):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
        """
        super().__init__(parties)

        self._element = SecaggElementTypes.SERVER_KEY

    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for server key secagg context element

        Returns:
            a tuple of a `context` and a `status` for the server key context element
        """
        # start dummy payload
        logger.info('PUT RESEARCHER SECAGG SERVER_KEY PAYLOAD HERE')
        time.sleep(1)
        context = { 'msg': 'Not implemented yet' }
        status = True
        # end dummy payload

        return context, status


class SecaggBiprimeContext(SecaggContext):
    """
    Handles a Secure Aggregation biprime context element on the researcher side.
    """

    def __init__(self, parties: List[str]):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
        """
        super().__init__(parties)

        self._element = SecaggElementTypes.BIPRIME

    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for biprime secagg context element

        Returns:
            a tuple of a `context` and a `status` for the biprime context element
        """
        # start dummy payload
        logger.info('PUT RESEARCHER SECAGG BIPRIME PAYLOAD HERE')
        time.sleep(3)
        context = { 'msg': 'Not implemented yet' }
        status = False
        # end dummy payload

        return context, status
