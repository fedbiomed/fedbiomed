# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation setup on the node"""
import inspect
from typing import List, Union
from abc import ABC, abstractmethod
from enum import Enum
import time
import random
import uuid

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, ComponentType, \
    REQUEST_PREFIX, TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeToNodeMessages
from fedbiomed.common.mpc_controller import MPCController

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg_manager import SKManager, BPrimeManager, DHManager
from fedbiomed.node.requests import send_nodes, PendingRequests


_CManager = CertificateManager(
    db_path=environ["DB_PATH"]
)


class SecaggBaseSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: Union[str, None] = None,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element
                is attached (empty string if no attached experiment)
            parties: List of parties participating in the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """

        if len(parties) < 3:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `parties` : {parties} : need  ' \
                'at least 3 parties for secure aggregation'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        if researcher_id != parties[0]:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `researcher_id` : {researcher_id} : ' \
                'needs to be the same as the first secagg party'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

        # assign argument values
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties
        self._element = None

        # to be set in subclasses
        self._secagg_manager = None

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
    def experiment_id(self) -> Union[str, None]:
        """Getter for `experiment_id`

        Returns:
            ID of the experiment to which this secagg context element is attached (empty string if no attached experiment)
        """
        return self._experiment_id

    @property
    def element(self) -> Enum:
        """Getter for secagg context element type

        Returns:
            secagg context element name
        """

        return self._element

    def _create_secagg_reply(self, message: str = '', success: bool = False) -> dict:
        """Create reply message for researcher after secagg setup phase.

        Args:
            message: text information concerning the secagg setup
            success: `True` if secagg element setup was successful, `False` otherway

        Returns:
            message to return to the researcher
        """

        # If round is not successful log error message
        if not success:
            logger.error(message)

        return {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'success': success,
            'msg': message,
            'command': 'secagg'
        }

    def setup(self) -> dict:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        # Caveat: don't test if a context exists for this `secagg_id` eg
        #   context = self._secagg_manager.get(self._secagg_id, self._experiment_id)
        # This is because we always need to update the (possibly) existing entry for this `secagg_id`
        # to address the case where the previous secagg setup succeeded on some nodes (which thus
        # created a context) and failed on some nodes. In that case, negotiating only on previously failed nodes
        # would result in either negotiation error or keep incoherent setting between nodes for
        # this `secagg_id`
        try:
            self._setup_specific()
        except FedbiomedError as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply("Can not setup secure aggregation context "
                                             f"on node for {self._secagg_id}.", False)
        except Exception as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply('Unexpected error occurred please '
                                             'report this to the node owner', False)

        return self._create_secagg_reply('Context element was successfully created on node', True)

    @abstractmethod
    def _setup_specific(self) -> None:
        """Service function for setting up a specific context element.
        """


class SecaggMpspdzSetup(SecaggBaseSetup):
    """
    Sets up a Secure Aggregation context element based on MPSPDZ on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: Union[str, None] = None,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element
                is attached (empty string if no attached experiment)
            parties: List of parties participating in the secagg context element setup
        """
        super().__init__(researcher_id, secagg_id, parties, experiment_id)

        # one controller per secagg object to prevent any file conflict
        self._MPC = MPCController(
            tmp_dir=environ["TMP_DIR"],
            component_type=ComponentType.NODE,
            component_id=environ["ID"]
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

        if not self._experiment_id or not isinstance(self._experiment_id, str):
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `experiment_id` must be a non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _setup_specific(self) -> None:
        """Service function for setting up the server key secagg context element.

        Raises:
            FedbiomedSecaggError: cannot read MPC computation results
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

        output = self._MPC.exec_shamir(
            party_number=self._parties.index(environ["ID"]),
            num_parties=len(self._parties),
            ip_addresses=ip_file
        )

        # Read output
        try:
            with open(output, "r") as file:
                key_share = file.read()
                file.close()
        except Exception as e:
            logger.debug("Can not open key share file written by MPC after executing MPC "
                         f"protocol. {e}. secagg_id: {self._secagg_id} file: {output}")

            # Message for researcher
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not access protocol output after applying multi party computation"
            )

        context = {'server_key': int(key_share)}
        self._secagg_manager.add(self._secagg_id, self._parties, context, self._experiment_id)
        logger.info(
            "Server key share successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")


class SecaggBiprimeSetup(SecaggMpspdzSetup):
    """
    Sets up a biprime Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: None = None):

        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: unused argument
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, parties, None)

        self._element = SecaggElementTypes.BIPRIME
        self._secagg_manager = BPrimeManager

        if experiment_id is not None:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `experiment_id` must be None'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _setup_specific(self) -> None:
        """Service function for setting up the biprime secagg context element.
        """
        # don't update an existing default biprime
        if not self._secagg_manager.is_default_biprime(self._secagg_id):
            # create a (currently dummy) context if it does not exist yet
            time.sleep(3)
            context = {
                'biprime': int(random.randrange(10**12)),   # dummy biprime
                'max_keysize': 0                            # prevent using the dummy biprime for real
            }
            logger.info("Not implemented yet, PUT SECAGG BIPRIME GENERATION PAYLOAD HERE, "
                        f"secagg_id='{self._secagg_id}'")

            # Currently, all biprimes can be used by all sets of parties.
            # TODO: add a mode where biprime is restricted for `self._parties`
            self._secagg_manager.add(self._secagg_id, None, context)
            logger.info(
                f"Biprime successfully created for node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")


class SecaggDhSetup(SecaggBaseSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: str,
            grpc_client: GrpcController,
            pending_requests: PendingRequests,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            experiment_id: ID of the experiment to which this secagg context element is attached
            parties: List of parties participating to the secagg context element setup
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, parties, experiment_id)

        self._element = SecaggElementTypes.DIFFIE_HELLMAN
        self._secagg_manager = DHManager
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests

        if not self._experiment_id or not isinstance(self._experiment_id, str):
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `experiment_id` must be a non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _setup_specific(self) -> None:
        """Service function for setting up the Diffie Hellman secagg context element.
        """
        # we know len(parties) >= 3 so len(other_nodes) >= 1
        other_nodes = [ e for e in self._parties[1:] if e != environ['NODE_ID'] ]

        # Key exchange with other nodes
        # TODO: replace `dummy` with real payload

        other_nodes_messages = []
        for node in other_nodes:
            other_nodes_messages += [
                NodeToNodeMessages.format_outgoing_message({
                    'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'dummy': f"KEY REQUEST INNER from {environ['NODE_ID']}",
                    'secagg_id': self._secagg_id,
                    'command': 'key-request'
                })
            ]

        logger.debug(f'Sending Diffie-Hellman setup for {self._secagg_id} to nodes: {other_nodes}')
        listener_id = send_nodes(
            self._grpc_client,
            self._pending_requests,
            self._researcher_id,
            other_nodes,
            other_nodes_messages,
        )
        all_received, messages = self._pending_requests.wait(listener_id, TIMEOUT_NODE_TO_NODE_REQUEST)

        logger.debug(f"Completed Diffie-Hellmann setup with success={all_received} "
                     f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}")
        if not all_received:
            nodes_no_answer = set(other_nodes) - set([m['node_id'] for m in messages])
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Some nodes did not answer during Diffie Hellman secagg "
                f"context setup: {nodes_no_answer}"
            )

        # Successful DH exchange with other ndoes
        context = { 'dummy': "tempo value to replace by LOM specific value"}
        self._secagg_manager.add(
            self._secagg_id,
            self._parties,
            context,
            self._experiment_id
        )
        logger.info(
            "Diffie Hellman secagg context successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")


class SecaggSetup:
    """Factory class for instantiating any type of node secagg context element setup class
    """

    element2class = {
        SecaggElementTypes.SERVER_KEY.name: SecaggServkeySetup,
        SecaggElementTypes.BIPRIME.name: SecaggBiprimeSetup,
        SecaggElementTypes.DIFFIE_HELLMAN.name: SecaggDhSetup
    }

    def __init__(self, element: int, **kwargs):
        """Constructor of the class
        """
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
                f"{ErrorNumbers.FB318.value}: Received bad request message: incorrect `element` {self._element}")

        try:
            args_to_init = {key: val for key, val in self.kwargs.items()
                            if key in inspect.signature(SecaggSetup.element2class[element.name].__init__).parameters}
            return SecaggSetup.element2class[element.name](**args_to_init)
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not instantiate secure aggregation setup with argument "
                f"{self.kwargs}. Error: {e}"
            )
