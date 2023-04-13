# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Secure Aggregation setup on the node"""
from typing import List, Union
from abc import ABC, abstractmethod
from enum import Enum
import time
import random

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import ErrorNumbers, SecaggElementTypes, ComponentType
from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.mpc_controller import MPCController
from fedbiomed.common.validator import Validator, ValidatorError
from fedbiomed.common.utils import matching_parties_servkey, matching_parties_biprime

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg_manager import SKManager, BPrimeManager


_CManager = CertificateManager(
    db_path=environ["DB_PATH"]
)


class BaseSecaggSetup(ABC):
    """
    Sets up a Secure Aggregation context element on the node side.
    """

    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str],
            job_id: Union[str, None] = None,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: ID of the job to which this secagg context element is attached (empty string if no attached job)
            sequence: unique sequence number of setup request
            parties: List of parties participating in the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        # check arguments
        self._v = Validator()
        for param, type in [(researcher_id, str), (secagg_id, str), (sequence, int)]:
            try:
                self._v.validate(param, type)
            except ValidatorError as e:
                errmess = f'{ErrorNumbers.FB318.value}: bad parameter `{param}` should be a {type}: {e}'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

        for param, name in [(researcher_id, 'researcher_id'), (secagg_id, 'secagg_id')]:
            if not param:
                errmess = f'{ErrorNumbers.FB318.value}: bad parameter `{name}` should not be empty string'
                logger.error(errmess)
                raise FedbiomedSecaggError(errmess)

        try:
            self._v.validate(parties, list)
            for p in parties:
                self._v.validate(p, str)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `parties` must be a list of strings: {e}'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)
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
        self._job_id = job_id
        self._sequence = sequence
        self._parties = parties
        self._element = None

        # one controller per secagg object to prevent any file conflict
        self._MPC = MPCController(
            tmp_dir=environ["TMP_DIR"],
            component_type=ComponentType.NODE,
            component_id=environ["ID"]
        )

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
    def job_id(self) -> Union[str, None]:
        """Getter for `job_id`

        Returns:
            ID of the job to which this secagg context element is attached (empty string if no attached job)
        """
        return self._job_id

    @property
    def sequence(self) -> str:
        """ Getter for `sequence`

        Returns:
            sequence number for this request
        """
        return self._sequence

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
            'sequence': self._sequence,
            'success': success,
            'msg': message,
            'command': 'secagg'
        }

    @abstractmethod
    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """

    def setup(self) -> dict:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """
        try:
            context = self._secagg_manager.get(self._secagg_id, self._job_id)
        except FedbiomedError as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply(
                f'Can not create secure aggregation context due to database error: {e}', False)
        except Exception as e:
            logger.debug(f"Can not create secure aggregation context due to database error: {e}")
            return self._create_secagg_reply('Can not create secure aggregation context', False)

        if context is None:
            try:
                self._setup_specific()
            except FedbiomedError as e:
                logger.debug(f"{e}")
                return self._create_secagg_reply(f'Can not setup secure aggregation it might be due to unregistered '
                                                 f'certificate for the federated setup. Please see error: {e}', False)
            except Exception as e:
                logger.debug(f"{e}")
                return self._create_secagg_reply('Unexpected error occurred please '
                                                 'report this to the node owner', False)
        else:
            # Need to ensure the read context has compatible parties with this element
            if (not self._matching_parties(context)):
                return self._create_secagg_reply(
                    f'Secagg element context for {self._secagg_id} exists but parties do not match',
                    False)

            message = f"Node secagg context element for {self._secagg_id} is already existing for job {self._job_id}"
            logger.info(message)
            return self._create_secagg_reply(message, True)

        return self._create_secagg_reply('Context element was successfully created on node', True)

    @abstractmethod
    def _setup_specific(self) -> None:
        """Service function for setting up a specific context element.
        """


class SecaggServkeySetup(BaseSecaggSetup):
    """
    Sets up a server key Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str],
            job_id: str,
    ):
        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: ID of the job to which this secagg context element is attached
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value 
        """
        super().__init__(researcher_id, secagg_id, sequence, parties, job_id)

        self._element = SecaggElementTypes.SERVER_KEY
        self._secagg_manager = SKManager

        if not self._job_id or not isinstance(self._job_id, str):
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `job_id` must be a non empty string'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """
        return matching_parties_servkey(context, self._parties)

    def _setup_specific(self) -> None:
        """Service function for setting up the server key secagg context element.
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
        self._secagg_manager.add(self._secagg_id, self._parties, context, self._job_id)
        logger.info(
            "Server key share successfully created for "
            f"node_id='{environ['NODE_ID']}' secagg_id='{self._secagg_id}'")


class SecaggBiprimeSetup(BaseSecaggSetup):
    """
    Sets up a biprime Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            sequence: int,
            parties: List[str],
            job_id: None = None):

        """Constructor of the class.

        Args:
            researcher_id: ID of the researcher that requests setup
            secagg_id: ID of secagg context element for this setup request
            job_id: unused argument
            sequence: unique sequence number of setup request
            parties: List of parties participating to the secagg context element setup

        Raises:
            FedbiomedSecaggError: bad argument type or value
        """
        super().__init__(researcher_id, secagg_id, sequence, parties, None)

        self._element = SecaggElementTypes.BIPRIME
        self._secagg_manager = BPrimeManager

        if job_id is not None:
            errmess = f'{ErrorNumbers.FB318.value}: bad parameter `job_id` must be None'
            logger.error(errmess)
            raise FedbiomedSecaggError(errmess)

    def _matching_parties(self, context: dict) -> bool:
        """Check if parties of given context are compatible with the secagg context element.

        Args:
            context: context to be compared with the secagg context element

        Returns:
            True if this context can be used with this element, False if not.
        """
        # Need to ensure that:
        # - either the existing element is not attached to specific parties (None)
        # - or existing element was established for the same parties or a superset of the parties
        return matching_parties_biprime(context, self._parties)

    def _setup_specific(self) -> None:
        """Service function for setting up the biprime secagg context element.
        """

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


class SecaggSetup:
    """Wrapper class for instantiating any type of node secagg context element setup class
    """

    element2class = {
        SecaggElementTypes.SERVER_KEY.name: SecaggServkeySetup,
        SecaggElementTypes.BIPRIME.name: SecaggBiprimeSetup
    }

    def __init__(self, element: int, **kwargs):
        """Constructor of the class
        """
        self._element = element
        self.kwargs = kwargs

    def __call__(self) -> BaseSecaggSetup:
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
            return SecaggSetup.element2class[element.name](**self.kwargs)
        except Exception as e:
            raise FedbiomedSecaggError(
                f"{ErrorNumbers.FB318.value}: Can not instantiate secure aggregation setup with argument "
                f"{self.kwargs}. Error: {e}"
            )
