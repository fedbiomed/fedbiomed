# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation management on the researcher"""
import importlib
import uuid
import concurrent.futures

from typing import Callable, List, Optional, Union, Tuple, Any, Dict
from abc import ABC, abstractmethod

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests, StopOnDisconnect, StopOnError, \
    StopOnTimeout

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, ComponentType
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError
from fedbiomed.common.mpc_controller import MPCController
from fedbiomed.common.secagg_manager import BaseSecaggManager, SecaggServkeyManager, SecaggDhManager
from fedbiomed.common.utils import (
    matching_parties_servkey,
    get_method_spec,
    get_default_biprime
)
from fedbiomed.common.message import Message, ResearcherMessages

_CManager = CertificateManager(
    db_path=environ["DB_PATH"]
)

# Instantiate one manager for each secagg element type
_SKManager = SecaggServkeyManager(environ['DB_PATH'])
_DHManager = SecaggDhManager(environ['DB_PATH'])


class SecaggContext(ABC):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """

    @abstractmethod
    def __init__(self, parties: List[str], experiment_id: str, secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            experiment_id: ID of the experiment to which this secagg context element is attached.
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

        self._secagg_id = secagg_id if secagg_id is not None else 'secagg_' + str(uuid.uuid4())
        self._parties = parties
        self._researcher_id = environ['ID']
        self._requests = Requests()
        self._status = False
        self._context = None
        self._experiment_id = experiment_id
        self._element: SecaggElementTypes

        # to be set in subclasses
        self._secagg_manager: Optional[BaseSecaggManager] = None

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
    def experiment_id(self) -> str:
        """Getter for secagg context element experiment_id

        Returns:
            secagg context element experiment_id
        """
        return self._experiment_id

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

    @abstractmethod
    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """

    def _create_payload(self, *args, **kwargs) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for a secagg context element

        Returns:
            a tuple of a `context` and a `status` for the context element
        """
        context = self._secagg_manager.get(self._secagg_id, self._experiment_id)
        if context is None:
            _, status = self._create_payload_specific(*args, **kwargs)
            context = self._secagg_manager.get(self._secagg_id, self._experiment_id)
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
    def _create_payload_specific(self) -> Tuple[Union[dict, None], bool]:
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
        status = self._secagg_manager.remove(self._secagg_id, self.experiment_id)
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
            msg: Message,
            can_set_status: bool,
            payload: Callable,
    ) -> bool:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round
            can_set_status: `True` if this action can result in a valid secagg context
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action

        Returns:
            True if secagg context element action could be done for all parties, False if at least
                one of the parties could not do the context element action.
        """
        # reset values in case `setup()` was already run (and fails during this new execution,
        # or this is a deletion)
        self._status = False
        self._context = None
        context, status = self._secagg_round_specific(msg, payload)

        success = all(status.values())
        if can_set_status and success:
            self._status = True
            self._context = context

        return success

    @abstractmethod
    def _secagg_round_specific(
            self,
            msg: Message,
            payload: Callable,
    ) -> Tuple[dict, dict[str, bool]]:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action

        Returns:
            A tuple of
                - a dict containing the context describing this secagg context element
                - a dict where key/values are node ids/boolean with success status for secagg on each party
        """

    def setup(self) -> bool:
        """Setup secagg context element on defined parties.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.
        """
        msg = ResearcherMessages.format_outgoing_message({
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'experiment_id': self._experiment_id,
            'parties': self._parties,
            'command': 'secagg',
        })

        return self._secagg_round(msg, True, self._create_payload)

    def delete(self) -> bool:
        """Delete secagg context element on defined parties.

        Returns:
            True if secagg context element could be deleted for all parties, False if at least
                one of the parties could not delete context element.
        """
        self._status = False
        self._context = None
        msg = ResearcherMessages.format_outgoing_message({
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'element': self._element.value,
            'experiment_id': self._experiment_id,
            'command': 'secagg-delete',
        })
        return self._secagg_round(msg, False, self._delete_payload)

    def save_state_breakpoint(self) -> Dict[str, Any]:
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
                "experiment_id": self._experiment_id,

            },
            "attributes": {
                "_status": self._status,
                "_context": self._context,
                "_researcher_id": self._researcher_id,
            }
        }
        return state

    @staticmethod
    def load_state_breakpoint(
            state: Dict[str, Any]
    ) -> 'SecaggContext':

        """
        Method for loading secagg state from breakpoint state

        Args:
            state: The state that will be loaded
        """

        # Get class
        cls = getattr(importlib.import_module(state["module"]), state["class"])

        # Validate experiment id
        spec = get_method_spec(cls)
        if 'experiment_id' in spec:
            secagg = cls(**state["arguments"])
        else:
            state["arguments"].pop('experiment_id')
            secagg = cls(**state["arguments"])

        for key, value in state["attributes"].items():
            setattr(secagg, key, value)

        return secagg


class SecaggMpspdzContext(SecaggContext):
    """
    Handles a Secure Aggregation context element based on MPSPDZ on the researcher side.
    """

    def __init__(self, parties: List[str], experiment_id: str, secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            experiment_id: ID of the experiment to which this secagg context element is attached.
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, experiment_id, secagg_id)

        if len(parties) < 3:
            raise FedbiomedSecaggError(
                f'{ErrorNumbers.FB415.value}: bad parameter `parties` : {parties} : need  '
                'at least 3 parties for secure aggregation')

        if environ['ID'] != parties[0]:
            raise FedbiomedSecaggError(
                f'{ErrorNumbers.FB415.value}: researcher should be the first party.'
            )

        # one controller per secagg object to prevent any file conflict
        self._mpc = MPCController(
            tmp_dir=environ["TMP_DIR"],
            component_type=ComponentType.RESEARCHER,
            component_id=environ["ID"]
        )

    def _secagg_round_specific(
            self,
            msg: Message,
            payload: Callable,
    ) -> Tuple[dict, dict[str, bool]]:
        """Negotiate secagg context element action with defined parties for a MPSPDZ key exchange.

        Args:
            msg: message sent to the parties during the round
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action

        Returns:
            A tuple of
                - a dict containing the context describing this secagg context element
                - a dict where key/values are node ids/boolean with success status for secagg on each party

        Raises:
            FedbiomedSecaggError: some parties did not answer before timeout or answered error
            FedbiomedSecaggError: local payload did not complete before timeout
        """
        # Federated request should stop if any error occurs
        policies = [StopOnDisconnect(timeout=30), StopOnError(), StopOnTimeout(timeout=120)]

        executer = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        context_future = executer.submit(payload)

        with self._requests.send(msg, self._parties[1:], policies) as fed_request:
            replies = fed_request.replies()
            errors = fed_request.errors()
            if fed_request.policy.has_stopped_any():
                self._mpc.kill()
                context_future.cancel()
                raise FedbiomedSecaggError("Request is not successful. Policy "
                                           f"report => {fed_request.policy.report()}. Errors => {errors}")

            status = {rep.node_id: rep.success for rep in replies.values()}

        try:
            context, status[self._researcher_id] = context_future.result(timeout=120)
        except TimeoutError:
            self._mpc.kill()
            context_future.cancel()
            raise FedbiomedSecaggError("Request is not successful, timeout on researcher payload.")

        return context, status


class SecaggServkeyContext(SecaggMpspdzContext):
    """
    Handles a Secure Aggregation server key context element on the researcher side.
    """

    def __init__(self, parties: List[str], experiment_id: str, secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            experiment_id: ID of the experiment to which this secagg context element is attached.
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, experiment_id, secagg_id)

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

    def _create_payload_specific(self) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for creating server key secagg context element

        Returns:
            A tuple of a `context` and a `status` for the server key context element
        """

        ip_file, _ = _CManager.write_mpc_certificates_for_experiment(
            path_certificates=self._mpc.mpc_data_dir,
            path_ips=self._mpc.tmp_dir,
            self_id=environ["ID"],
            self_ip=environ["MPSPDZ_IP"],
            self_port=environ["MPSPDZ_PORT"],
            self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
            self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
            parties=self._parties
        )

        try:
            output = self._mpc.exec_shamir(
                party_number=0,  # 0 stands for server/aggregator
                num_parties=len(self._parties),
                ip_addresses=ip_file
            )
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: Can not execute MPC protocol. {e}") from e

        # Read output
        try:
            with open(output, "r", encoding="UTF-8") as file:
                server_key = file.read()
                file.close()
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: Can not read server key from created after "
                f"MPC execution. {e}") from e

        biprime = get_default_biprime()
        context = {'server_key': int(server_key.strip()), 'biprime': biprime}
        self._secagg_manager.add(self._secagg_id, self._parties, context, self._experiment_id)
        logger.debug(
            f"Server key successfully created for researcher_id='{environ['ID']}' "
            f"secagg_id='{self._secagg_id}'")

        return context, True



class SecaggDHContext(SecaggContext):
    """
    Handles a Secure Aggregation Diffie Hellman context element on the researcher side.
    """

    def __init__(self, parties: List[str], experiment_id: str, secagg_id: Union[str, None] = None):
        """Constructor of the class.

        Args:
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            experiment_id: ID of the experiment to which this secagg context element is attached.
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(parties, experiment_id, secagg_id)

        if len(parties) < 2:
            raise FedbiomedSecaggError(
                f'{ErrorNumbers.FB415.value}: LOM, bad parameter `parties` : {parties} : need '
                'at least 2 nodes for secure aggregation')


        self._element = SecaggElementTypes.DIFFIE_HELLMAN
        self._secagg_manager = _DHManager

    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """
        return matching_parties_servkey(context, self._parties)

    def _create_payload_specific(self, context) -> Tuple[Union[dict, None], bool]:
        """Researcher payload for creating Diffie Hellman  secagg context element

        Args:
            context: specific secagg element context

        Returns:
            A tuple of a `context` and a `status` for the server key context element
        """
        self._secagg_manager.add(self._secagg_id, self._parties, context, self._experiment_id)

        logger.debug(
            f"Diffie Hellman context successfully created for researcher_id='{environ['ID']}' "
            f"secagg_id='{self._secagg_id}'")

        return context, True

    def _secagg_round_specific(
            self,
            msg: Message,
            payload: Callable,
    ) -> Tuple[dict, dict[str, bool]]:
        """Negotiate secagg context element action with defined parties for a MPSPDZ key exchange.

        Args:
            msg: message sent to the parties during the round
            payload: function that holds researcher side payload for this round. Needs to return
                a tuple of `context` and `status` for this action

        Returns:
            A tuple of
                - a dict containing the context describing this secagg context element
                - a dict where key/values are node ids/boolean with success status for secagg on each party

        Raises:
            FedbiomedSecaggError: some parties did not answer before timeout or answered error
            FedbiomedSecaggError: local payload did not complete before timeout
        """
        # Federated request should stop if any error occurs
        policies = [StopOnDisconnect(timeout=30), StopOnError(), StopOnTimeout(timeout=120)]
        # Ensure that reply from each node was received
        status = {node_id: False for node_id in self._parties}

        with self._requests.send(msg, self._parties, policies) as fed_request:
            replies = fed_request.replies()
            errors = fed_request.errors()

            status = {rep.node_id: rep.success for rep in replies.values()}
            status[self._researcher_id] = not errors and not fed_request.policy.has_stopped_any()

        # no cryptographic material on the researcher side in this case, no specific context element to save
        context = {}
        if all(status.values()):
            # only create context if previous steps succeeded
            _, status_researcher_payload = payload(context)
            status[self._researcher_id] = status[self._researcher_id] and status_researcher_payload

        # Current implementation does not try to clean entries on successful nodes if some nodes failed (`not status`)

        return context, status
