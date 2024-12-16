# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation setup on the node"""
import random
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union

from fedbiomed.common.constants import (
    REQUEST_PREFIX,
    ErrorNumbers,
    SecaggElementTypes,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedSecaggError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    AdditiveSSharingRequest,
    AdditiveSSSetupReply,
    ErrorMessage,
    KeyRequest,
    Message,
    SecaggReply,
)
from fedbiomed.common.secagg import (
    AdditiveSecret,
    AdditiveShare,
    AdditiveShares,
    DHKey,
    DHKeyAgreement,
)
from fedbiomed.common.synchro import EventWaitExchange
from fedbiomed.common.utils import get_default_biprime
from fedbiomed.node.requests import send_nodes, NodeToNodeRouter
from fedbiomed.node.secagg_manager import SecaggManager
from fedbiomed.transport.controller import GrpcController


class SecaggBaseSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """
    _element: SecaggElementTypes
    _REPLY_CLASS: Message = SecaggReply
    _min_num_parties: int = 3

    def __init__(
        self,
        db: str,
        node_id: str,
        researcher_id: str,
        secagg_id: str,
        parties: List[str],
        experiment_id: Union[str, None],
    ):
        """Constructor of the class.

        Args:
            db: Path to database file the node.
            node_id: ID of the node.
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            parties: List of parties participating in the secagg context element setup
            experiment_id: ID of the experiment to which this secagg context element
                is attached

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        errmess: str = ""
        if len(parties) < self._min_num_parties:
            errmess = (
                f"{ErrorNumbers.FB318.value}: bad parameter `parties` : {parties} : need  "
                f"at least {self._min_num_parties} parties for secure aggregation, but got "
                f"{len(parties)}"
            )

        if researcher_id is None:
            errmess = (
                f"{ErrorNumbers.FB318.value}: argument `researcher_id` must be a non-None value"
            )

        if errmess:
            # if one of the above condition is met, raise error
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # assign argument values
        self._secagg_manager = SecaggManager(db, self._element.value)()
        self._node_id = node_id
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties

    @property
    def researcher_id(self) -> str:
        """Getter for `researcher_id`

        Returns:
            researcher unique ID
        """
        return self._researcher_id

    @property
    def secagg_id(self) -> str:
        """Getter for `secagg_id`

        Returns:
            secagg context element unique ID
        """
        return self._secagg_id

    @property
    def experiment_id(self) -> str:
        """Getter for `experiment_id`

        Returns:
            ID of the experiment to which this secagg context element is attached
        """
        return self._experiment_id

    @property
    def element(self) -> Enum:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """

        return self._element

    def _create_secagg_reply(
        self, success: bool = True, message: str = "", **kwargs
    ) -> Message:
        """Create reply message for researcher after secagg setup phase.

        Args:
            success: `True` if secagg element setup was successful, `False` otherway
            message: text information concerning the secagg setup
        Returns:
            message to return to the researcher
        """
        common = {"node_id": self._node_id, "researcher_id": self._researcher_id}
        # If round is not successful log error message
        if not success:
            logger.error(message)
            return ErrorMessage(**{**common, "extra_msg": message})

        return self._REPLY_CLASS(
            **{
                **common,
                "secagg_id": self._secagg_id,
                "success": success,
                "msg": message,
                **kwargs,
            }
        )

    def setup(self) -> Message:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        try:
            return self._setup_specific()
        except FedbiomedError as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply(
                False,
                "Can not setup secure aggregation context "
                f"on node for {self._secagg_id}. {e}",
            )

    @abstractmethod
    def _setup_specific(self) -> Message:
        """Service function for setting up a specific context element."""


class _SecaggNN(SecaggBaseSetup):

    def __init__(
            self,
            *args,
            n2n_router: NodeToNodeRouter,
            grpc_client: GrpcController,
            pending_requests: EventWaitExchange,
            controller_data: EventWaitExchange,
            **kwargs,
    ):
        """Constructor of the class.

        Args:
            n2n_router: object managing node to node messages
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for passing data to the node controller
            *args: Please see [SecaggBaseSetup]
            **kwargs: Please see [SecaggBaseSetup]
        Raises:
            FedbiomedSecaggError: bad argument type or value
        """

        super().__init__(*args, **kwargs)

        # self._secagg_manager = SKManager
        self._n2n_router = n2n_router
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests
        self._controller_data = controller_data

    def _send(self, nodes, messages: List[Message]):
        """Sends given message to nodes"""

        return send_nodes(
            self._n2n_router,
            self._grpc_client,
            self._pending_requests,
            self._researcher_id,
            nodes,
            messages,
            raise_if_not_all_received=True,
        )


class SecaggServkeySetup(_SecaggNN):
    """Secure aggregation setup phase for ServerKey generation on the node side"""

    _REPLY_CLASS: Message = AdditiveSSSetupReply
    _key_bit_length: int = 2040
    _min_num_parties: int = 2
    _element = SecaggElementTypes.SERVER_KEY

    # def __init__(self, share, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def _setup_specific(self) -> Message:

        other_nodes = list(filter(lambda x: x != self._node_id, self._parties))
        # Create secret user key and its shares
        seed = random.SystemRandom()
        sk = seed.getrandbits(self._key_bit_length)
        user_key = AdditiveSecret(sk)
        shares = user_key.split(len(self._parties)).to_list()

        # The last share is the share for the node who executes the request
        my_share = AdditiveShare(shares.pop(-1))
        party_shares = dict(zip(other_nodes, shares))
        self._controller_data.event(self._secagg_id, {"shares": party_shares})

        requests = [
            AdditiveSSharingRequest(
                request_id=REQUEST_PREFIX + str(uuid.uuid4()),
                node_id=self._node_id,
                dest_node_id=node,
                secagg_id=self._secagg_id,
            )
            for (i, node) in enumerate(other_nodes)
        ]

        all_received, messages = self._send(other_nodes, requests)
        sum_shares = my_share + sum(
            AdditiveShares([AdditiveShare(reply.share) for reply in messages])
        )

        logger.debug(
            f"Completed Serverkey secret sharing setup with success={all_received} "
            f"node_id='{self._node_id}' secagg_id='{self._secagg_id}"
        )

        biprime = get_default_biprime()
        context = {"server_key": int(sk), "biprime": int(biprime)}
        self._secagg_manager.add(
            self._secagg_id, self._parties, context, self._experiment_id
        )

        logger.info(
            "Server key share successfully created for "
            f"node_id='{self._node_id}' secagg_id='{self._secagg_id}'"
        )

        return self._create_secagg_reply(share=sum_shares.value)


class SecaggDHSetup(_SecaggNN):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """

    _element = SecaggElementTypes.DIFFIE_HELLMAN
    _min_num_parties: int = 2

    def _setup_specific(self) -> Message:
        """Service function for setting up the Diffie Hellman secagg context element."""
        # we know len(parties) >= 3 so len(other_nodes) >= 1
        other_nodes = [e for e in self._parties if e != self._node_id]

        local_keypair = DHKey()
        key_agreement = DHKeyAgreement(
            node_u_id=self._node_id,
            node_u_dh_key=local_keypair,
            session_salt=bytes(self._secagg_id, "utf-8"),
        )

        # Make public key available for requests received from other nodes for this `secagg_id`
        self._controller_data.event(
            self._secagg_id, {"public_key": local_keypair.export_public_key()}
        )

        # Request public key from other nodes
        other_nodes_messages = []
        for node in other_nodes:
            other_nodes_messages += [
                KeyRequest(
                    request_id=REQUEST_PREFIX + str(uuid.uuid4()),
                    node_id=self._node_id,
                    dest_node_id=node,
                    secagg_id=self._secagg_id,
                )
            ]

        logger.debug(
            f"Sending Diffie-Hellman setup for {self._secagg_id} to nodes: {other_nodes}"
        )

        _, messages = self._send(other_nodes, other_nodes_messages)

        # At this point: successful DH exchange with other nodes
        context = {
            m.get_param("node_id"): key_agreement.agree(
                m.get_param("node_id"), m.get_param("public_key")
            )
            for m in messages
        }

        self._secagg_manager.add(
            self._secagg_id, self._parties, context, self._experiment_id
        )

        logger.info(
            "Diffie Hellman secagg context successfully created for "
            f"node_id='{self._node_id}' secagg_id='{self._secagg_id}'"
        )

        return self._create_secagg_reply()


class SecaggSetup:
    """Factory class for instantiating any type of node secagg context element setup class"""

    element2class = {
        SecaggElementTypes.SERVER_KEY.name: SecaggServkeySetup,
        SecaggElementTypes.DIFFIE_HELLMAN.name: SecaggDHSetup,
    }

    def __init__(self, element: int, **kwargs):
        """Constructor of the class"""
        self._element = element
        self.kwargs = kwargs

    def __call__(self) -> SecaggBaseSetup:
        """Instantiate a node secagg context element setup class.

        Returns:
            a new secagg context element object
        """

        if self._element in [m.value for m in SecaggElementTypes]:
            element = SecaggElementTypes(self._element)
        else:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Received bad request message: "
                "incorrect `element` {self._element}"
            )

        try:
            return SecaggSetup.element2class[element.name](**self.kwargs)
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not instantiate secure aggregation "
                f"setup with argument {self.kwargs}. Error: {e}"
            ) from e
