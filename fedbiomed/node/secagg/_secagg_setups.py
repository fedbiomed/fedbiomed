# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation setup on the node"""
import random
import inspect
from typing import List
from abc import ABC, abstractmethod
from enum import Enum
import uuid

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import (
    ErrorNumbers,
    SecaggElementTypes,
    ComponentType,
    REQUEST_PREFIX,
)
from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    Message,
    SecaggReply,
    ErrorMessage,
    AdditiveSSharingReply,
    AdditiveSSharingRequest,
    AdditiveSSSetupReply,
    NodeToNodeMessages,
)
from fedbiomed.common.mpc_controller import MPCController
from fedbiomed.common.secagg import DHKey, DHKeyAgreement
from fedbiomed.common.synchro import EventWaitExchange
from fedbiomed.common.utils import get_default_biprime
from fedbiomed.common.secagg import AdditiveSecret, AdditiveShares, AdditiveShare

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg_manager import SKManager, DHManager, SecaggManager
from fedbiomed.node.requests import send_nodes


_CManager = CertificateManager(db_path=environ["DB_PATH"])


class SecaggBaseSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    _REPLY_CLASS: Message = SecaggReply
    _min_num_parties: int = 3

    def __init__(
        self,
        researcher_id: str,
        secagg_id: str,
        parties: List[str],
        experiment_id: str,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element
                is attached
            parties: List of parties participating in the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        errmess: str = ""
        if len(parties) < self._min_num_parties:
            errmess = (
                f"{ErrorNumbers.FB318.value}: bad parameter `parties` : {parties} : need  "
                f"at least {self._min_num_parties} parties for secure aggregation, but got {len(parties)}"
            )

        if researcher_id is None:
            errmess = f"{ErrorNumbers.FB318.value}: argument `researcher_id` must be a non-None value"

        if errmess:
            # if one of the above condition is met, raise error
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # assign argument values
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties
        self._element: SecaggElementTypes = None

        # to be set in subclasses
        self._secagg_manager: SecaggManager = None

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
        common = {"node_id": environ["ID"], "researcher_id": self._researcher_id}
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

    def setup(self) -> dict:
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


class SecaggMpspdzSetup(SecaggBaseSetup):
    """
    Sets up a Secure Aggregation context element based on MPSPDZ on the node side.
    """

    def __init__(
        self,
        researcher_id: str,
        secagg_id: str,
        parties: List[str],
        experiment_id: str,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element
                is attached
            parties: List of parties participating in the secagg context element setup
        """

        if parties and researcher_id != parties[0]:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: bad parameter `researcher_id` : {researcher_id} : "
                f"needs to be the same as the first secagg party `parties[0]`: {parties[0]}"
            )

        super().__init__(researcher_id, secagg_id, parties, experiment_id)

        # one controller per secagg object to prevent any file conflict
        self._mpc = MPCController(
            tmp_dir=environ["TMP_DIR"],
            component_type=ComponentType.NODE,
            component_id=environ["ID"],
        )


class SecaggServkeySetup(SecaggMpspdzSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """

    def __init__(
        self,
        researcher_id: str,
        secagg_id: str,
        parties: List[str],
        experiment_id: str,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element is attached
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, parties, experiment_id)

        self._element = SecaggElementTypes.SERVER_KEY
        self._secagg_manager = SKManager

    def _setup_specific(self) -> Message:
        """Service function for setting up the server key secagg context element.

        Raises:
            FedbiomedSecaggError: cannot read MPC computation results
        """

        ip_file, _ = _CManager.write_mpc_certificates_for_experiment(
            path_certificates=self._mpc.mpc_data_dir,
            path_ips=self._mpc.tmp_dir,
            self_id=environ["ID"],
            self_ip=environ["MPSPDZ_IP"],
            self_port=environ["MPSPDZ_PORT"],
            self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
            self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
            parties=self._parties,
        )

        output = self._mpc.exec_shamir(
            party_number=self._parties.index(environ["ID"]),
            num_parties=len(self._parties),
            ip_addresses=ip_file,
        )

        # Read output
        try:
            with open(output, "r", encoding="UTF-8") as file:
                key_share = file.read()

        except Exception as e:
            logger.debug(
                "Can not open key share file written by MPC after executing MPC "
                f"protocol. {e}. secagg_id: {self._secagg_id} file: {output}"
            )

            # Message for researcher
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not access protocol output after applying multi party computation"
            )

        biprime = get_default_biprime()
        context = {"server_key": int(key_share), "biprime": int(biprime)}
        self._secagg_manager.add(
            self._secagg_id, self._parties, context, self._experiment_id
        )
        logger.info(
            "Server key share successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'"
        )

        return self._create_secagg_reply()


class _SecaggNN(SecaggBaseSetup):

    def __init__(
        self,
        *args,
        grpc_client: GrpcController,
        pending_requests: EventWaitExchange,
        controller_data: EventWaitExchange,
        **kwargs,
    ):
        """Constructor of the class.

        Args:
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for passing data to the node controller
            *args**kwargs: Please see [SecaggBaseSetup]
        Raises:
            FedbiomedSecaggError: bad argument type or value
        """

        super().__init__(*args, **kwargs)

        self._secagg_manager = SKManager
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests
        self._controller_data = controller_data

    def _send(self, nodes, messages: List[Message]):
        """Sends given message to nodes"""

        return send_nodes(
            self._grpc_client,
            self._pending_requests,
            self._researcher_id,
            nodes,
            messages,
            raise_if_not_all_received=True,
        )


class SecaggKeySetup(_SecaggNN):
    """Secure aggregation setup phase for ServerKey generation on the node side"""

    _REPLY_CLASS: Message = AdditiveSSSetupReply
    _key_bit_length: int = 2040
    _min_num_parties: int = 2
    _element = SecaggElementTypes.SERVER_KEY

    def _setup_specific(self) -> Message:

        other_nodes = filter(lambda x: x != environ["ID"], self._parties)

        # Create secret user key and its shares
        seed = random.SystemRandom()
        sk = seed.getrandbits(self._key_bit_length)
        user_key = AdditiveSecret(sk)
        shares = user_key.split(len(self._parties)).to_list()

        # The last share is the share for the node who executes the request
        my_share = shares.pop(-1)
        self._controller_data.event(self._secagg_id, {"shares": shares})

        messages = [
            AdditiveSSharingRequest(
                request_id=REQUEST_PREFIX + str(uuid.uuid4()),
                node_id=environ["NODE_ID"],
                dest_node_id=node,
                secagg_id=self._secagg_id,
            )
            for (i, node) in enumerate(other_nodes)
        ]

        all_received, messages = self._send(other_nodes, messages)
        sum_shares = my_share + sum(
            AdditiveShares([AdditiveShare(reply.share) for reply in messages])
        )

        logger.debug(
            f"Completed Serverkey secret sharing setup with success={all_received} "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}"
        )

        biprime = get_default_biprime()
        context = {"server_key": int(sk), "biprime": int(biprime)}
        self._secagg_manager.add(
            self._secagg_id, self._parties, context, self._experiment_id
        )
        logger.info(
            "Server key share successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'"
        )

        return self._create_secagg_reply(share=sum_shares)


class SecaggDHSetup(SecaggBaseSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """

    _min_num_parties: int = 2

    def __init__(
        self,
        researcher_id: str,
        secagg_id: str,
        parties: List[str],
        experiment_id: str,
        grpc_client: GrpcController,
        pending_requests: EventWaitExchange,
        controller_data: EventWaitExchange,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element is attached
            parties: List of parties participating to the secagg context element setup
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
            controller_data: object for passing data to the node controller

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, parties, experiment_id)

        self._element = SecaggElementTypes.DIFFIE_HELLMAN
        self._secagg_manager = DHManager
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests
        self._controller_data = controller_data

    def _setup_specific(self) -> Message:
        """Service function for setting up the Diffie Hellman secagg context element."""
        # we know len(parties) >= 3 so len(other_nodes) >= 1
        other_nodes = [e for e in self._parties if e != environ["NODE_ID"]]

        local_keypair = DHKey()
        key_agreement = DHKeyAgreement(
            node_u_id=environ["NODE_ID"],
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
                NodeToNodeMessages.format_outgoing_message(
                    {
                        "request_id": REQUEST_PREFIX + str(uuid.uuid4()),
                        "node_id": environ["NODE_ID"],
                        "dest_node_id": node,
                        "secagg_id": self._secagg_id,
                        "command": "key-request",
                    }
                )
            ]

        logger.debug(
            f"Sending Diffie-Hellman setup for {self._secagg_id} to nodes: {other_nodes}"
        )
        all_received, messages = send_nodes(
            self._grpc_client,
            self._pending_requests,
            self._researcher_id,
            other_nodes,
            other_nodes_messages,
        )
        # Nota: don't clean `self._controller_data.remove(secagg_id)` when finished.
        # Rely on automatic cleaning after timeout.
        # This node received all replies, but some nodes may still be querying this node.

        logger.debug(
            f"Completed Diffie-Hellmann setup with success={all_received} "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}"
        )
        if not all_received:
            nodes_no_answer = set(other_nodes) - set(
                [m.get_param("node_id") for m in messages]
            )
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Some nodes did not answer during Diffie Hellman secagg "
                f"context setup: {nodes_no_answer}"
            )

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

        self._secagg_manager.get(self._secagg_id, self._experiment_id)
        # At this point: successfully negotiated and save secagg context
        logger.info(
            "Diffie Hellman secagg context successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'"
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
                f"{ErrorNumbers.FB318.value}: Received bad request message: incorrect `element` {self._element}"
            )

        try:
            args_to_init = {
                key: val
                for key, val in self.kwargs.items()
                if key
                in inspect.signature(
                    SecaggSetup.element2class[element.name].__init__
                ).parameters
            }
            return SecaggSetup.element2class[element.name](**args_to_init)
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not instantiate secure aggregation setup with argument "
                f"{self.kwargs}. Error: {e}"
            )
