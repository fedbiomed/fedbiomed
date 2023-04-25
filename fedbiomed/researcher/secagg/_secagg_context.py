# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation management on the researcher"""
import importlib
import uuid
from typing import Callable, List, Union, Tuple, Any, Dict
from abc import ABC, abstractmethod
import time
import random

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, ComponentType
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError
from fedbiomed.common.mpc_controller import MPCController
from fedbiomed.common.secagg_manager import SecaggServkeyManager, SecaggBiprimeManager
from fedbiomed.common.utils import matching_parties_servkey, matching_parties_biprime, get_method_spec


_CManager = CertificateManager(
    db_path=environ["DB_PATH"]
)

# Instantiate one manager for each secagg element type
_SKManager = SecaggServkeyManager(environ['DB_PATH'])
_BPrimeManager = SecaggBiprimeManager(environ['DB_PATH'])


class SecaggContext(ABC):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """

    def __init__(self, parties: List[str], job_id: Union[str, None], secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            job_id: ID of the job to which this secagg context element is attached.
                None means the element is not attached to a specific job
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        self._v = Validator()

        self._v.register("nonempty_str_or_none", self._check_secagg_id_type, override=True)
        try:
            self._v.validate(secagg_id, "nonempty_str_or_none")
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `secagg_id` must be a None or non-empty string: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

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

        if environ['ID'] != parties[0]:
            raise FedbiomedSecaggError(
                f'{ErrorNumbers.FB415.value}: researcher should be the first party.'
            )

        self._secagg_id = secagg_id if secagg_id is not None else 'secagg_' + str(uuid.uuid4())
        self._parties = parties
        self._researcher_id = environ['ID']
        self._requests = Requests()
        self._status = False
        self._context = None
        self._job_id = None

        # set job ID using setter to validate
        self.set_job_id(job_id)

        # one controller per secagg object to prevent any file conflict
        self._MPC = MPCController(
            tmp_dir=environ["TMP_DIR"],
            component_type=ComponentType.RESEARCHER,
            component_id=environ["ID"]
        )

        # to be set in subclasses
        self._secagg_manager = None

    @staticmethod
    def _check_secagg_id_type(value) -> bool:
        """Check if argument is None or a non-empty string

        Args:
            value: argument to check.

        Returns:
            True if argument matches constraint, False if it does not.
        """
        return value is None or (isinstance(value, str) and bool(value))

    @property
    def parties(self) -> str:
        """Getter for secagg parties

        Returns:
            Parties that participates secure aggregation
        """
        return self._parties

    @property
    def secagg_id(self) -> str:
        """Getter for secagg context element ID 

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    @property
    def job_id(self) -> Union[str, None]:
        """Getter for secagg context element job_id

        Returns:
            secagg context element job_ib (or None if no job_id is attached to the element)
        """
        return self._job_id

    @property
    def status(self) -> bool:
        """Getter for secagg context element status

        Returns:
            `True` if secagg context element exists, `False` otherwise
        """
        return self._status

    # alternative: define method in subclass to have specific return type
    @property
    def context(self) -> Union[dict, None]:
        """Getter for secagg context element content

        Returns:
            secagg context element, or `None` if it doesn't exist
        """
        return self._context

    def set_job_id(self, job_id: Union[str, None]) -> None:
        """Setter for secagg context element job_id

        Args:
            job_id: ID of the job to which this secagg context element is attached.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """

        if not isinstance(job_id, (str, type(None))):
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `job_id` must be a str or None if the ' \
                      f'context is set for biprime.'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._job_id = job_id

    @abstractmethod
    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """

    def _payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for a secagg context element

        Returns:
            a tuple of a `context` and a `status` for the biprime context element
        """
        context = self._secagg_manager.get(self._secagg_id, self._job_id)

        if context is None:
            _, status = self._payload_create()
            context = self._secagg_manager.get(self._secagg_id, self._job_id)
        else:
            # Need to ensure the read context has compatible parties with this element
            if not self._matching_parties(context):
                logger.error(
                    f"{ErrorNumbers.FB415.value}: secagg context for {self._secagg_id} exists "
                    f"but parties do not match")
                status = False
            else:
                logger.debug(
                    f"Secagg context for {self._secagg_id} is already existing on researcher "
                    f"researcher_id='{environ['ID']}'")
                status = True

        return context, status

    @abstractmethod
    def _payload_create(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for creating secagg context element, specific to a context element type.

        Returns:
            a tuple of a `context`, and a boolean `status` for the context element.
        """

    def _delete_payload(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for secagg context element deletion

        Returns:
            a tuple of None (no context after deletion) and
                a boolean (True if payload succeeded for this element)
        """
        status = self._secagg_manager.remove(self._secagg_id, self.job_id)
        if status:
            logger.debug(
                f"Context element successfully deleted for researcher_id='{environ['ID']}' "
                f"secagg_id='{self._secagg_id}'")
        else:
            logger.error(
                f"{ErrorNumbers.FB415.value}: No such context element secagg_id={self._secagg_id} "
                f"on researcher researcher_id='{environ['ID']}'")

        return None, status

    def _secagg_round(
            self,
            msg: dict,
            command: str,
            can_set_status: bool,
            payload: Callable,
            timeout: float = 0
    ) -> bool:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round
            command: reply command expected from the parties
            can_set_status: `True` if this action can result in a valid secagg context
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action
            timeout: maximum time waiting for answers from other nodes, after completing the setup locally.
                It does not include the time for the local setup payload.
                Defaults to `environ['TIMEOUT']` if unset or equals 0.

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

        # FIXME: There are scenarios where key-share are calculated but it is lost on the
        # researcher side
        sequence = {}
        for node in self._parties[1:]:
            sequence[node] = self._requests.send_message(msg, node, add_sequence=True)
        status = {}

        # basic implementation: synchronous payload on researcher, then read answers from other parties
        context, status[self._researcher_id] = payload()

        # `timeout` covers only the time waiting for answers from other nodes
        start_time = time.time()

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

        return return_value

    def setup(
            self,
            timeout: float = 0
    ) -> bool:
        """Setup secagg context element on defined parties.

        Args:
            timeout: maximum time waiting for answers from other nodes, after completing the setup locally.
                It does not include the time for the local setup payload.
                Defaults to `environ['TIMEOUT']` if unset or equals 0.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.

        Raises:
            FedbiomedSecaggError: bad argument type
        """
        if isinstance(timeout, int):
            timeout = float(timeout)  # accept int (and bool...)
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

    def delete(
            self,
            timeout: float = 0
    ) -> bool:
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
            timeout = float(timeout)  # accept int (and bool...)
        try:
            self._v.validate(timeout, float)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `timeout`: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._status = False
        self._context = None
        msg = {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'job_id': self._job_id,
            'command': 'secagg-delete',
        }
        return self._secagg_round(msg, 'secagg-delete', False, self._delete_payload, timeout)

    def save_state(self) -> Dict[str, Any]:
        """Method for saving secagg state for saving breakpoints

        Returns:
            The state of the secagg
        """
        # `_v` and `_requests` dont need to be savec (properly initiated in constructor)
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                "secagg_id": self._secagg_id,
                "parties": self._parties,
                "job_id": self._job_id,

            },
            "attributes": {
                "_status": self._status,
                "_context": self._context,
                "_researcher_id": self._researcher_id,
            }
        }
        return state

    @staticmethod
    def load_state(
            state: Dict[str, Any]
    ) -> 'SecaggContext':

        """
        Method for loading secagg state from breakpoint state

        Args:
            state: The state that will be loaded
        """

        # Get class
        cls = getattr(importlib.import_module(state["module"]), state["class"])

        # Validate job id
        spec = get_method_spec(cls)
        if 'job_id' in spec:
            secagg = cls(**state["arguments"])
        else:
            state["arguments"].pop('job_id')
            secagg = cls(**state["arguments"])

        for key, value in state["attributes"].items():
            setattr(secagg, key, value)

        return secagg


class SecaggServkeyContext(SecaggContext):
    """
    Handles a Secure Aggregation server key context element on the researcher side.
    """

    def __init__(self, parties: List[str], job_id: str, secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            job_id: ID of the job to which this secagg context element is attached.
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, job_id, secagg_id)

        if not self._job_id:
            errmess = f'{ErrorNumbers.FB415.value}: bad parameter `job_id` must be non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        self._element = SecaggElementTypes.SERVER_KEY
        self._secagg_manager = _SKManager

    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """
        return matching_parties_servkey(context, self._parties)

    def _payload_create(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for creating server key secagg context element

        Returns:
            A tuple of a `context` and a `status` for the server key context element
        """

        ip_file, _ = _CManager.write_mpc_certificates_for_experiment(
            path_certificates=self._MPC.mpc_data_dir,
            path_ips=self._MPC.tmp_dir,
            self_id=environ["ID"],
            self_ip=environ["MPSPDZ_IP"],
            self_port=environ["MPSPDZ_PORT"],
            self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
            self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
            parties=self._parties
        )

        try:
            output = self._MPC.exec_shamir(
                party_number=0,  # 0 stands for server/aggregator
                num_parties=len(self._parties),
                ip_addresses=ip_file
            )
        except Exception as e:
            raise FedbiomedSecaggError(f"{ErrorNumbers.FB415.value}: Can not execute MPC protocol. {e}")

        # Read output
        try:
            with open(output, "r") as file:
                server_key = file.read()
                file.close()
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: Can not read server key from created after MPC execution. {e}"
            )

        context = {'server_key': int(server_key.strip())}
        self._secagg_manager.add(self._secagg_id, self._parties, context, self._job_id)
        logger.debug(
            f"Server key successfully created for researcher_id='{environ['ID']}' "
            f"secagg_id='{self._secagg_id}'")

        return context, True


class SecaggBiprimeContext(SecaggContext):
    """
    Handles a Secure Aggregation biprime context element on the researcher side.
    """

    def __init__(self, parties: List[str], secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating to the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, None, secagg_id)

        self._element = SecaggElementTypes.BIPRIME
        self._secagg_manager = _BPrimeManager

    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """
        return matching_parties_biprime(context, self._parties)

    def _payload_create(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for creating biprime secagg context element

        Returns:
            a tuple of a `context` and a `status` for the biprime context element
        """
        # start dummy payload
        time.sleep(3)
        context = {
            'biprime': int(random.randrange(10**12)),   # dummy biprime
            'max_keysize': 0                            # prevent using the dummy biprime for real
        }
        logger.info('Not yet implemented, PUT RESEARCHER SECAGG BIPRIME PAYLOAD HERE')

        # Currently, all biprimes can be used by all sets of parties.
        # TODO: add a mode where biprime is restricted for `self._parties`
        self._secagg_manager.add(self._secagg_id, None, context)
        logger.debug(
            f"Biprime successfully created for researcher_id='{environ['ID']}' secagg_id='{self._secagg_id}'")
        # end dummy payload

        return context, True
