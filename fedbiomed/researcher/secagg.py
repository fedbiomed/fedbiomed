# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation management on the researcher"""
import uuid
from typing import Callable, List, Union, Tuple, Any, Dict
from abc import ABC, abstractmethod
import time

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests


class SecaggContext(ABC):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """

    def __init__(self, parties: List[str], job_id: str):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            job_id: ID of the job to which this secagg context element is attached.
                Empty string means the element is not attached to a specific job

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        self._v = Validator()
        try:
            self._v.validate(parties, list)
            for p in parties:
                self._v.validate(p, str)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `parties` must be a list of strings: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        if len(parties) < 3:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `parties` : {parties} : need  ' \
                'at least 3 parties for secure aggregation'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._secagg_id = 'secagg_' + str(uuid.uuid4())
        self._parties = parties
        self._researcher_id = environ['RESEARCHER_ID']
        self._requests = Requests()
        self._status = False
        self._context = None

        self.set_job_id(job_id)

    def secagg_id(self) -> str:
        """Getter for secagg context element ID 

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    def job_id(self) -> str:
        """Getter for secagg context element job_id

        Returns:
            secagg context element job_ib (or empty string if no job_id is attached to the element)
        """
        return self._job_id

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

    def set_job_id(self, job_id: str) -> None:
        """Setter for secagg context element job_id

        Args:
            job_id: ID of the job to which this secagg context element is attached.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        try:
            self._v.validate(job_id, str)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `job_id` must be a str: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._job_id = job_id

    @abstractmethod
    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for secagg context element

        Returns:
            a tuple of a `context`, and a boolean `status` for the context element.
        """
        # abstract method may contain some code in the future, so let's keep this command
        pass

    def _delete_payload(self) ->  Tuple[Union[dict, None], bool]:
        """Researcher payload for secagg context element deletion

        Returns:
            a tuple of None (no context after deletion) and True (payload succeeded for this element)
        """
        return None, True


    def _secagg_round(self, msg: dict, command: str, can_set_status: bool, payload: Callable, timeout: float = 0) -> bool:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round
            command: reply command expected from the parties
            can_set_status: `True` if this action can result in a valid secagg context
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action
            timeout: maximum duration for the negotiation phase. Defaults to `environ['TIMEOUT']` if unser
                or equals 0.

        Returns:
            True if secagg context element action could be done for all parties, False if at least
                one of the parties could not do the context element action.

        Raises:
            FedbiomedSecaggError: some parties did not answer before timeout
            FedbiomedSecaggError: received a reply for a non-party to the negotiation
        """
        # reset values in case `setup()` was already run (and fails during this new execution,
        # or this is a deletion)
        return_value = False
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
                    continue
                if resp['secagg_id'] != self._secagg_id:
                    logger.debug(
                        f"Unexpected secagg reply: expected `secagg_id` {self._secagg_id}"
                        f" and received {resp['secagg_id']}")
                    continue
                if resp['node_id'] not in self._parties[1:]:
                    errmess = f'{ErrorNumbers.FB415.value}: received message from node "{resp["node_id"]}"' \
                        'which is not a party of secagg "{self._secagg_id}"'
                    logger.error(errmess)
                    raise FedbiomedSecaggError(errmess)
                if resp['sequence'] != sequence[resp['node_id']]:
                    logger.debug(
                        f"Out of sequence secagg reply: expected `sequence` {sequence[resp['node_id']]}"
                        f" and received {resp['sequence']}"
                    )
                    continue

                # this answer belongs to current secagg context setup
                status[resp['node_id']] = resp['success']

            if set(status.keys()) == set(self._parties):
                break

        if not set(status.keys()) == set(self._parties):
            # case where some parties did not answer
            # self._status = False
            # self._context = None
            absent = list(set(self._parties) - set(status.keys()))
            errmess = f'{ErrorNumbers.FB415.value}: some parties did not answer before timeout {absent}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
        else:
            return_value = all(status.values())
            if can_set_status and return_value:
                self._status = True
                self._context = context
            # else:
            #    self._status = False
            #    self._context = None

        return return_value

    def setup(self, timeout: float = 0) -> bool:
        """Setup secagg context element on defined parties.

        Args:
            timeout: maximum duration for the setup phase. Defaults to `environ['TIMEOUT']` if unset
                or equals 0.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.

        Raises:
            FedbiomedSecaggError: bad argument type
        """
        if isinstance(timeout, int):
            timeout = float(timeout)    # accept int (and bool...)
        try:
            self._v.validate(timeout, float)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `timeout`: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        msg = {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'job_id': self._job_id,
            'parties': self._parties,
            'command': 'secagg',
        }
        return self._secagg_round(msg, 'secagg', True, self._payload, timeout)

    def delete(self, timeout: float = 0) -> bool:
        """Delete secagg context element on defined parties.

        Args:
            timeout: maximum duration for the deletion phase. Defaults to `environ['TIMEOUT']` if unset
                or equals 0.

        Returns:
            True if secagg context element could be deleted for all parties, False if at least
                one of the parties could not delete context element.

        Raises:
            FedbiomedSecaggError: bad argument type
        """
        if isinstance(timeout, int):
            timeout = float(timeout)    # accept int (and bool...)
        try:
            self._v.validate(timeout, float)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `timeout`: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        if self._status:
            msg = {
                'researcher_id': self._researcher_id,
                'secagg_id': self._secagg_id,
                'element': self._element.value,
                'job_id': self._job_id,
                'command': 'secagg-delete',
            }
            return self._secagg_round(msg, 'secagg-delete', False, self._delete_payload, timeout)
        else:
            self._context = None   # should already be the case
            return False

    def save_state(self) -> Dict[str, Any]:
        """Method for saving secagg state for saving breakpoints

        Returns:
            The state of the secagg
        """
        # `_v` and `_requests` dont need to be savec (properly initiated in constructor)
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "secagg_id": self._secagg_id,
            "parties": self._parties,
            "job_id": self._job_id,
            "researcher_id": self._researcher_id,
            "status": self._status,
            "context": self._context
        }
        return state

    def load_state(self, state: Dict[str, Any] = None, **kwargs):
        """
        Method for loading secagg state from breakpoint state

        Args:
            state: The state that will be loaded
        """
        self._secagg_id = state['secagg_id']
        self._parties = state['parties']
        self._job_id = state['job_id']
        self._researcher_id = state['researcher_id']
        self._status = state['status']
        self._context = state['context']


class SecaggServkeyContext(SecaggContext):
    """
    Handles a Secure Aggregation server key context element on the researcher side.
    """

    def __init__(self, parties: List[str], job_id: str):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            job_id: ID of the job to which this secagg context element is attached.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, job_id)

        if not self._job_id:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `job_id` must be non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._element = SecaggElementTypes.SERVER_KEY

    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for server key secagg context element

        Returns:
            a tuple of a `context` and a `status` for the server key context element
        """
        # start dummy payload
        time.sleep(1)
        logger.info('PUT RESEARCHER SECAGG SERVER_KEY PAYLOAD HERE')
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

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, '')

        self._element = SecaggElementTypes.BIPRIME

    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for biprime secagg context element

        Returns:
            a tuple of a `context` and a `status` for the biprime context element
        """
        # start dummy payload
        time.sleep(3)
        logger.info('PUT RESEARCHER SECAGG BIPRIME PAYLOAD HERE')
        context = { 'msg': 'Not implemented yet' }
        status = True
        # end dummy payload

        return context, status
