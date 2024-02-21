# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import random

from abc import ABCMeta, abstractmethod
from typing import List, Union, Dict, Any, Optional


from ._secagg_context import SecaggServkeyContext, SecaggBiprimeContext
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.secagg import JLSCrypter, FlamingoCrypter
from fedbiomed.common.logger import logger


class SecureAggregationSchemes:
    """Secure aggregation schemes"""
    JOYE_LIBERT = 'JOYELIBERT'
    FLAMINGO = "FLAMINGO"


class SecureAggregation:
    """Secure aggregation controller of researcher component.

    This class is responsible for;

    - setting up the context for Joye-Libert secure aggregation
    - Applying secure aggregation after receiving encrypted model parameters from nodes

    Attributes:
        clipping_range: Clipping range that will be used for quantization of model
            parameters on the node side.

        _biprime: Biprime-key context setup instance.
        _parties: Nodes and researcher that participates federated training
        _job_id: ID of the current Job launched by the experiment.
        _servkey: Server-key context setup instance.
        _secagg_crypter: Secure aggregation encrypter and decrypter to decrypt encrypted model
            parameters.
        _secagg_random: Random float generated tobe sent to node to validate secure aggregation
            after aggregation encrypted parameters.
    """

    def __init__(
            self,
            active: bool = True,
            clipping_range: Union[None, int] = None,
            scheme: str = 'jls'
    ) -> None:
        """Class constructor

        Assigns default values for attributes

        Args:
            active: True if secure aggregation is activated for the experiment
            clipping_range: Clipping range that will be used for quantization of model
                parameters on the node side. The default will be
                [`VEParameters.CLIPPING_RANGE`][fedbiomed.common.constants.VEParameters].
                The default value will be automatically set on the node side.

        Raises:
            FedbiomedSecureAggregationError: bad argument type
        """

        if not isinstance(active, bool):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: The argument `active` should be  bool of type, "
                f"but got {type(active)} "
            )

        if clipping_range is not None and \
                (not isinstance(clipping_range, int) or isinstance(clipping_range, bool)):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Clipping range should be None or an integer, "
                f"but got not {type(clipping_range)}"
            )

        self._parties = None
        self._job_id = None
        self._active: bool = active

        self.scheme = scheme
        self.clipping_range: Optional[int] = clipping_range

        if scheme == 'jls':
            self.aggregator = JLSAggregator()
        elif scheme == 'flamingo':
            self.aggregator = FlamingoAggregator()
        else:
            raise FedbiomedSecureAggregationError(
                f"Undefined secure aggragation scheme {scheme}"
            )

    @property
    def parties(self) -> Union[List[str], None]:
        """Gets secagg parties

        Returns:
            List of secagg parties if it exists, or None
        """
        return self._parties

    @property
    def job_id(self) -> Union[str, None]:
        """Gets secagg associated job_id

        Returns:
            str of associated job_id if it exists, or None
        """
        return self._job_id

    @property
    def active(self) -> bool:
        """Gets secagg activation status

        Returns:
            bool, True if secagg is activated
        """
        return self._active

    def activate(self, status) -> bool:
        """Set activate status of secure aggregation

        Returns:
            Status of secure aggregation True if it is activated
        """

        if not isinstance(status, bool):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: The argument `status` for activation should be True or False, "
                f"but got {type(status)} "
            )

        self._active = status

        return self._active

    def train_arguments(self) -> Dict:
        """Gets train arguments for secagg train request

        Returns:
            Arguments that is going tobe attached to the experiment.
        """
        return self.aggregator.arguments()

    def setup(self,
              parties: List[str],
              job_id: str,
              force: bool = False):
        """Setup secure aggregation instruments.

        Requires setting `parties` and `job_id` if they are not set in previous secagg
        setups. It is possible to execute without any argument if SecureAggregation
        has already `parties` and `job_id` defined. This feature provides researcher
        execute `secagg.setup()` if any connection issue

        Args:
            parties: Parties that participates secure aggregation
            job_id: The id of the job of experiment
            force: Forces secagg setup even context is already existing

        Returns:
            Status of setup

        Raises
            FedbiomedSecureAggregationError: Invalid argument type
        """

        return self.aggregator.setup(parties, job_id, force)


    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            params: Dict[str, List[int]]
    ) -> List[float]:
        """Aggregates given model parameters

        Args:
            round_: current training round number
            total_sample_size: sum of number of samples used by all nodes
            params: model parameters from the participating nodes
        Returns:
            Aggregated parameters

        Raises:
            FedbiomedSecureAggregationError: secure aggregation context not properly configured
            FedbiomedSecureAggregationError: secure aggregation computation error
        """

        return self.aggregator.aggregate(
            round_, total_sample_size, params
        )

    def save_state_breakpoint(self) -> Dict[str, Any]:
        """Saves state of the secagg

        Returns:
            The secagg state to be saved
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                'active': self._active,
                'clipping_range': self.clipping_range,
                'scheme': self.scheme
            },
            "attributes": {
                "aggregator": self.aggregator.save_state(),
                "_job_id": self._job_id,
                "_parties": self._parties
            }
        }

        return state

    @classmethod
    def load_state_breakpoint(
            cls,
            state: Dict
    ) -> 'SecureAggregation':
        """Create a `SecureAggregation` object from a saved state

        Args:
            state: saved state to restore in the created object

        Returns:
            The created `SecureAggregation` object
        """

        secagg = cls(**state["arguments"])

        # Load aggregator state
        state["attributes"]["aggregator"] = secagg.aggregator.load_state(
            state=state["attributes"]["aggregator"])

        # Set attributes
        for name, val in state["attributes"].items():
            setattr(secagg, name, val)

        return secagg


class SecureAggregator(metaclass=ABCMeta):

    def __init__(
        self,
        clipping_range: Optional[int] = None
    ) -> None:
        """Constructs abstract-base aggregator class

        Args:
          clipping_range: Clipping range used for quantization
        """
        self._crypter = None
        self._parties = []
        self.clipping_range = clipping_range

    @abstractmethod
    def setup(
        self,
        parties: List[str],
        job_id: str,
        force: bool = False
    ):
        """Setup secure aggregation instruments.

        Requires setting `parties` and `job_id` if they are not set in previous secagg
        setups. It is possible to execute without any argument if SecureAggregation
        has already `parties` and `job_id` defined. This feature provides researcher
        execute `secagg.setup()` if any connection issue

        Args:
            parties: Parties that participates secure aggregation
            job_id: The id of the job of experiment
            force: Forces secagg setup even context is already existing

        Returns:
            Status of setup

        Raises
            FedbiomedSecureAggregationError: Invalid argument type
        """
        if not isinstance(parties, list):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `parties` list but got {type(parties)}"
            )

        if not isinstance(job_id, str):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `job_id` string but got {type(parties)}"
            )

        self._configure_round(parties, job_id)


    def _configure_round(
        self,
        parties,
        job_id
    ) -> bool:
        """Configuries secure aggregation elements for the round

        This method checks the round state and creates secagg context element if
        not existing or re-instantiates if the state of the round has changes in cases of
        adding new nodes to the FL training

        Args:
            parties: Nodes that participates federated training
            job_id: The id of the job of experiment
        """
        if not self._parties or self._job_id != job_id:
            self._set_secagg_contexts(parties, job_id)

        elif set(self._parties) != set(parties):
            logger.info(f"Parties of the experiment has changed. Re-creating secure "
                        f"aggregation context creation for the experiment {self._job_id}")
            self._set_secagg_contexts(parties)

    @abstractmethod
    def _set_secagg_contexts(
        self,
        parties: List[str],
        job_id: Union[str, None] = None
    ) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `job_id` and `parties` are set

        Args:
            parties: Parties that participates secure aggregation
            job_id: The id of the job of experiment
        """

        self._parties = parties

        # Updates job id if it is provided
        if job_id is not None:
            self._job_id = job_id

    @abstractmethod
    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            params: Dict[str, List[int]],
    ) -> List[float]:
        """Aggregates"""

    @abstractmethod
    def save_state(self):
        """Saves secure aggregator state"""

    @abstractmethod
    def load_state(self):
        """Creates secure aggregator instance from given state

        Args:
            state: Dict contains aggregator state
        """


class JLSAggregator(SecureAggregator):
    """Aggregator class that uses Joye-Libert algorithm"""

    def __init__(
        self,
        clipping_range: int = None
    ) -> None:
        """Constructs Joye-Libert aggregator"""

        super().__init__(clipping_range)

        self._crypter = None
        self._biprime = None
        self._servkey = None

    def _set_secagg_contexts(
        self,
        parties: List[str],
        job_id: Union[str, None] = None
    ) -> None:
        """Sets secure aggregation contexts"""
        # Call base class
        super()._set_secagg_contexts(parties, job_id)

        self._biprime = SecaggBiprimeContext(
            parties=self._parties,
            secagg_id='default_biprime0'
        )

        self._servkey = SecaggServkeyContext(
            parties=self._parties,
            job_id=self._job_id
        )

    @property
    def biprime(self) -> Union[None, SecaggBiprimeContext]:
        """Gets biprime object

        Returns:
            Biprime object, None if biprime is not setup
        """
        return self._biprime

    @property
    def servkey(self) -> Union[None, SecaggServkeyContext]:
        """Gets servkey object

        Returns:
            Servkey object, None if servkey is not setup
        """
        return self._servkey


    def arguments(self) -> Dict:
        """Gets train arguments for secagg train request

        Returns:
            Arguments that is going tobe attached to the experiment.
        """
        return {'secagg_servkey_id': self._servkey.secagg_id if self._servkey is not None else None,
                'secagg_biprime_id': self._biprime.secagg_id if self._biprime is not None else None,
                'secagg_clipping_range': self.clipping_range,
                'secagg_scheme': SecureAggregationSchemes.JOYE_LIBERT}

    def setup(
        self,
        parties: List[str],
        job_id: str,
        force: bool = False
    ) -> bool:
        """Sets up Joye-Libert secure aggregation context"""

        # Apply common actions
        super().setup(parties, job_id, force)

        if self._biprime is None or self._servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: server key or biprime contexts is not fully configured."
            )

        if not self._biprime.status or force:
            self._biprime.setup()

        if not self._servkey.status or force:
            self._servkey.setup()

        # Set crypter class
        self._crypter = JLSCrypter(
            n_parties=len(parties),
            biprime=self._biprime.context["context"]["biprime"],
            key=self._servkey.context["context"]["server_key"]
        )

        return True

    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            params: Dict[str, List[int]],
    ) -> List[float]:
        """Aggregates"""

        if self._biprime is None or self._servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, one of Biprime or Servkey context is"
                f"not configured. Please setup secure aggregation before the aggregation.")

        if not self._biprime.status or not self._servkey.status:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, one of Biprime or Servkey context is"
                f"not set properly")

        vectors = [p for _, p in params.items()]
        aggregated_params = self._crypter.aggregate(
            round_=round_,
            vectors=vectors,
            weight_deminator=total_sample_size,
            clipping_range=self.clipping_range
        )

        return aggregated_params

    def save_state(self):
        """Saves states of the JLS aggregator"""

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                'clipping_range': self.clipping_range,
            },
            "attributes": {
                "_biprime": self._biprime.save_state_breakpoint() if self._biprime is not None else None,
                "_servkey": self._servkey.save_state_breakpoint() if self._servkey is not None else None,
            }
        }

        return state

    @classmethod
    def load_state(
        cls,
        state: Dict
    ) -> 'JLSAggregator':
        """Load self class from given state object """

        secagg = cls(**state["arguments"])

        if state["attributes"]["_biprime"] is not None:
            state["attributes"]["_biprime"] = SecaggBiprimeContext. \
                load_state_breakpoint(state=state["attributes"]["_biprime"])

        if state["attributes"]["_servkey"] is not None:
            state["attributes"]["_servkey"] = SecaggServkeyContext. \
                load_state_breakpoint(state=state["attributes"]["_servkey"])

        # Set attributes
        for name, val in state["attributes"].items():
            setattr(secagg, name, val)

        return secagg


class FlamingoAggregator(SecureAggregator):

    def __init__(self):
        pass

    def setup(
        self,
        parties: List[str],
        job_id: str,
        force: bool = False
    ) -> bool:
        """Sets up secure aggregation context for Flamingo

        TODO: Implement DH key exhange request among the node here
        """

        # TODO: After setup instantiate FlamingoCrypter
        pass

    def _configure_round(self, parties, job_id):
        """Method to execute at each round training"""
        # TODO: This method is already provided by base class SecureAggregator
        # it checks whether parties has changed after the previous round
        # please see SecureAggregator._configure_round and extend if necessary
        pass

    def _set_secagg_contexts(self, parties, job_id):
        """Sets contexts for Flamingo"""

        # TODO: Here the context for FLamingo should be setup
        # different than Joye-Libert, no need to keep servkey or biprime
        # Only creating secagg_id and lunching DH key exhange will be enough

    def arguments(self) -> Dict:
        """Returns arguments for secure aggregation request

        # IMPORTANT: Renamed from train_arguments to arguments since
          secure aggregation is not only used for training
        """
        return {'secagg_scheme': SecureAggregationSchemes.FLAMINGO}



    def aggregate(
        self,
        round_: int,
        total_sample_size: int,
        params: Dict[str, List[int]],
    ):
        """Aggregator given params/vectors

        TODO: params can be renamed as vectors to be more generic
        """
        pass


    def save_state(self) -> Dict:
        """Saves state of the FlamingoAggregator"""
        pass


    def load_state(self) -> 'FlamingoAggregator':
        """Instantiates a new FlamingoAggregator from given state"""
        pass
