"""
This file is originally part of Fed-BioMed
SPDX-License-Identifier: Apache-2.0


Secure aggregation context setup module  to be executed on the
server/researcher side

"""

import importlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    AdditiveSSSetupRequest,
    Message,
    SecaggDeleteRequest,
    SecaggRequest,
)
from fedbiomed.common.secagg import AdditiveShare, AdditiveShares
from fedbiomed.common.secagg_manager import (
    BaseSecaggManager,
    SecaggDhManager,
    SecaggServkeyManager,
)
from fedbiomed.common.utils import (
    get_default_biprime,
    get_method_spec,
    matching_parties_servkey,
)
from fedbiomed.common.validator import Validator, ValidatorError
from fedbiomed.researcher.config import config
from fedbiomed.researcher.requests import (
    Requests,
    StopOnDisconnect,
    StopOnError,
    StopOnTimeout,
)


class SecaggContext(ABC):
    """
    Handles a Secure Aggregation context element on the researcher side.
    """
    _REQUEST_SETUP = SecaggRequest
    _REQUEST_DELETE = SecaggDeleteRequest
    _min_num_parties: int

    @abstractmethod
    def __init__(
        self,
        researcher_id: str,
        parties: List[str],
        experiment_id: str,
        secagg_id: Union[str, None] = None
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that context will be created for.
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

        self._v.register(
            "nonempty_str_or_none", self._check_secagg_id_type, override=True
        )
        try:
            self._v.validate(secagg_id, "nonempty_str_or_none")
        except ValidatorError as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: bad parameter "
                "`secagg_id` must be a None or non-empty string: {e}"
            ) from e

        try:
            self._v.validate(parties, list)
            for p in parties:
                self._v.validate(p, str)
        except ValidatorError as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: bad parameter "
                f"`parties` must be a list of strings: {e}"
            ) from e

        self._researcher_id = researcher_id
        self._db = config.vars["DB"]
        self._secagg_id = (
            secagg_id if secagg_id is not None else "secagg_" + str(uuid.uuid4())
        )
        self._parties = parties
        self._requests = Requests(config)
        self._status = False
        self._context = None
        self._experiment_id = experiment_id
        self._element: SecaggElementTypes

        # to be set in subclasses
        self._secagg_manager: Optional[BaseSecaggManager] = None

    def _raise_if_missing_parties(self, parties: List[str]):
        if len(parties) < self._min_num_parties:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB415.value}: {self._element.value}, bad parameter "
                f"`parties` : {parties} : need at least {self._min_num_parties} "
                "nodes for secure aggregation"
            )

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

    def _register(self, context) -> Dict:
        """Researcher payload for creating Additive Secret Sharing secagg context element

        Args:
            context: specific secagg element context

        Returns:
            A tuple of a `context` and a `status` for the server key context element
        """
        self._secagg_manager.add(
            self._secagg_id, self._parties, context, self._experiment_id
        )

        logger.debug(
            f"Secure aggregation context successfully created/registered for"
            f"researcher_id='{self._researcher_id}' secagg_id='{self._secagg_id}'"
        )

        return context

    def _launch_request(self, request: Message):
        """Launches and collects request responses

        Args:
            request: A request message
        """

        # Federated request should stop if any error occurs
        policies = [
            StopOnDisconnect(timeout=10),
            StopOnError(),
            StopOnTimeout(timeout=60),
        ]

        with self._requests.send(request, self._parties, policies) as fed_request:
            replies = fed_request.replies()
            errors = fed_request.errors()

            status = not errors and not fed_request.policy.has_stopped_any()

        if not status:
            raise FedbiomedSecaggError(
                f"Secure aggregation request: `{request.__name__}` has failed on one or more "
                f"nodes: {fed_request.policy.report()}, Errors: {errors}"
            )

        return replies

    @abstractmethod
    def secagg_round(
        self,
        request: Message,
    ) -> bool:
        """Negotiate secagg context element action with defined parties.

        Args:
            msg: message sent to the parties during the round

        Returns:
            True if secagg context element action could be done for all parties, False if at least
                one of the parties could not do the context element action.
        """

    def setup(self) -> Dict:
        """Setup secagg context element on defined parties.

        Returns:
            True if secagg context element could be setup for all parties, False if at least
                one of the parties could not setup context element.
        """
        context = self._secagg_manager.get(self._secagg_id, self._experiment_id)
        if context and matching_parties_servkey(context, self._parties):
            logger.info(
                f"{ErrorNumbers.FB415.value}: secagg context for {self._secagg_id} exists"
            )
            self._context = context['context']
        else:
            request = self._REQUEST_SETUP(
                researcher_id=self._researcher_id,
                secagg_id=self._secagg_id,
                element=self._element.value,
                experiment_id=self._experiment_id,
                parties=self._parties,
            )

            self._context = self.secagg_round(request)

        self._status = True

        return self._status


    def delete(self) -> bool:
        """Delete secagg context element on defined parties.

        Returns:
            True if secagg context element could be deleted for all parties, False if at least
                one of the parties could not delete context element.
        """
        self._status = False
        self._context = None
        request = self._REQUEST_DELETE(
            researcher_id=self._researcher_id,
            secagg_id=self._secagg_id,
            element=self._element.value,
            experiment_id=self._experiment_id,
        )

        _ = self._launch_request(request)

        status = self._secagg_manager.remove(self._secagg_id, self.experiment_id)
        if status:
            logger.debug(
                f"Context element successfully deleted for researcher_id='{self._researcher_id}' "
                f"secagg_id='{self._secagg_id}'"
            )
        else:
            logger.error(
                f"{ErrorNumbers.FB415.value}: No such context element secagg_id={self._secagg_id} "
                f"on researcher researcher_id='{self._researcher_id}'"
            )

        return True

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
                "researcher_id": self._researcher_id,
            },
            "attributes": {
                "_status": self._status,
                "_context": self._context,
            },
        }
        return state

    @staticmethod
    def load_state_breakpoint(state: Dict[str, Any]) -> "SecaggContext":
        """
        Method for loading secagg state from breakpoint state

        Args:
            state: The state that will be loaded
        """

        # Get class
        cls = getattr(importlib.import_module(state["module"]), state["class"])

        secagg = cls(**state["arguments"])
        for key, value in state["attributes"].items():
            setattr(secagg, key, value)

        return secagg


class SecaggServkeyContext(SecaggContext):

    # Nodes  + researcher
    _REQUEST_SETUP = AdditiveSSSetupRequest
    _min_num_parties: int = 2

    def __init__(
        self,
        researcher_id,
        parties: List[str],
        experiment_id: str,
        secagg_id: str | None = None,
    ):
        """Constructs key context class"""
        super().__init__(researcher_id, parties, experiment_id, secagg_id)

        self._element = SecaggElementTypes.SERVER_KEY
        self._raise_if_missing_parties(parties)
        self._secagg_manager = SecaggServkeyManager(self._db)


    def secagg_round(
        self,
        request: Message,
    ) -> Tuple[dict, dict[str, bool]]:
        """Negotiate secagg context element action with defined parties for secret sharing.

        Args:
            request: message sent to the parties during the round

        Returns:
            A tuple of
                - a dict containing the context describing this secagg context element
                - a dict where key/values are node ids/boolean with success status
                    for secagg on each party
        """

        replies = self._launch_request(request)
        servkey: int = AdditiveShares([AdditiveShare(rep.share) for
            rep in replies.values()]).reconstruct()

        biprime = get_default_biprime()
        context = {'server_key': -servkey, 'biprime': biprime}
        return self._register(context)


class SecaggDHContext(SecaggContext):
    """
    Handles a Secure Aggregation Diffie Hellman context element on the researcher side.
    """
    _REQUEST = SecaggRequest
    _min_num_parties: int = 2

    def __init__(
        self,
        researcher_id: str,
        parties: List[str],
        experiment_id: str,
        secagg_id: Union[str, None] = None
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that context will be created for.
            parties: list of parties participating in the secagg context element setup, named
                by their unique id (`node_id`, `researcher_id`).
                There must be at least 3 parties, and the first party is this researcher
            experiment_id: ID of the experiment to which this secagg context element is attached.
            secagg_id: optional secagg context element ID to use for this element.
                Default is None, which means a unique element ID will be generated.

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, parties, experiment_id, secagg_id)
        self._element = SecaggElementTypes.DIFFIE_HELLMAN
        self._raise_if_missing_parties(parties)
        self._secagg_manager = SecaggDhManager(self._db)

    def secagg_round(
        self,
        request: Message,
    ) -> Tuple[dict, dict[str, bool]]:
        """Negotiate secagg context element action with defined parties for DH key exchange.

        Args:
            request: message sent to the parties during the round

        Returns:
            A tuple of
                - a dict containing the context describing this secagg context element
                - a dict where key/values are node ids/boolean with success status for
                    secagg on each party

        """

        _ = self._launch_request(request)
        context = {}
        return self._register(context)

