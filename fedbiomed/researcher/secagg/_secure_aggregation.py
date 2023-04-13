# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import random
from typing import List, Union, Dict, Any, Optional

from ._secagg_context import SecaggServkeyContext, SecaggBiprimeContext
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.secagg import SecaggCrypter
from fedbiomed.common.logger import logger


class SecureAggregation:
    """Secure aggregation controller of researcher component.

    This class is responsible for;

    - setting up the context for Joye-Libert secure aggregation
    - Applying secure aggregation after receiving encrypted model parameters from nodes

    Attributes:
        timeout: Maximum time waiting for answers from other nodes for each secagg context
            element (server key and biprime).
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
            timeout: int = 10,
            clipping_range: Union[None, int] = None,
    ) -> None:
        """Class constructor

        Assigns default values for attributes

        Args:
            active: True if secure aggregation is activated for the experiment
            timeout: Maximum time waiting for answers from other nodes for each
                secagg context element (server key and biprime). Thus total secagg
                setup is at most twice the `timeout`, plus the local setup payload
                execution time for server key and biprime. Defaults to `environ['TIMEOUT']`
                if unset or equals 0.
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

        if not isinstance(timeout, int):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: The argument `timeout` should be  an integer, "
                f"but got {type(active)} "
            )

        if clipping_range is not None and \
                (not isinstance(clipping_range, int) or isinstance(clipping_range, bool)):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Clipping range should be None or an integer, "
                f"but got not {type(clipping_range)}"
            )

        self.timeout: int = timeout
        self.clipping_range: Optional[int] = clipping_range

        self._active: bool = active
        self._parties: Optional[List[str]] = None
        self._job_id: Optional[str] = None
        self._servkey: Optional[SecaggServkeyContext] = None
        self._biprime: Optional[SecaggBiprimeContext] = None
        self._secagg_random: Optional[float] = None
        self._secagg_crypter: SecaggCrypter = SecaggCrypter()

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
        return {'secagg_servkey_id': self._servkey.secagg_id if self._servkey is not None else None,
                'secagg_biprime_id': self._biprime.secagg_id if self._biprime is not None else None,
                'secagg_random': self._secagg_random,
                'secagg_clipping_range': self.clipping_range}

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

        if not isinstance(parties, list):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `parties` list but got {type(parties)}"
            )

        if not isinstance(job_id, str):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `job_id` string but got {type(parties)}"
            )

        self._configure_round(parties, job_id)

        if self._biprime is None or self._servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: server key or biprime contexts is not fully configured."
            )

        if not self._biprime.status or force:
            self._biprime.setup(timeout=self.timeout)

        if not self._servkey.status or force:
            self._servkey.setup(timeout=self.timeout)

        return True

    def _set_secagg_contexts(self, parties: List[str], job_id: Union[str, None] = None) -> None:
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

        # TODO: support other options than using `default_biprime0`
        self._biprime = SecaggBiprimeContext(
            parties=self._parties,
            secagg_id='default_biprime0'
        )

        self._servkey = SecaggServkeyContext(
            parties=self._parties,
            job_id=self._job_id
        )

    def _configure_round(
            self,
            parties: List[str],
            job_id: str
    ) -> None:
        """Configures secure aggregation for each round.

        This method checks the round state and creates secagg context element if
        not existing or re-instantiates if the state of the round has changes in cases of
        adding new nodes to the FL training

        Args:
            parties: Nodes that participates federated training
            job_id: The id of the job of experiment
        """

        # For each round it generates new secagg random float
        self._secagg_random = round(random.uniform(0, 1), 3)

        if self._parties is None or self._job_id != job_id:
            self._set_secagg_contexts(parties, job_id)

        elif set(self._parties) != set(parties):
            logger.info(f"Parties of the experiment has changed. Re-creating secure "
                        f"aggregation context creation for the experiment {self._job_id}")
            self._set_secagg_contexts(parties)

    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            model_params: Dict[str, List[int]],
            encryption_factors: Union[Dict[str, List[int]], None] = None,
    ) -> List[float]:
        """Aggregates given model parameters

        Args:
            round_: current training round number
            total_sample_size: sum of number of samples used by all nodes
            model_params: model parameters from the participating nodes
            encryption_factors: encryption factors from the participating nodes

        Returns:
            Aggregated parameters

        Raises:
            FedbiomedSecureAggregationError: secure aggregation context not properly configured
            FedbiomedSecureAggregationError: secure aggregation computation error
        """

        if self._biprime is None or self._servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, one of Biprime or Servkey context is"
                f"not configured. Please setup secure aggregation before the aggregation.")

        if not self._biprime.status or not self._servkey.status:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, one of Biprime or Servkey context is"
                f"not set properly")

        biprime = self._biprime.context["context"]["biprime"]
        key = self._servkey.context["context"]["server_key"]

        num_nodes = len(model_params)

        aggregate = functools.partial(self._secagg_crypter.aggregate,
                                      current_round=round_,
                                      num_nodes=num_nodes,
                                      key=key,
                                      total_sample_size=total_sample_size,
                                      biprime=biprime,
                                      clipping_range=self.clipping_range)

        # Validate secure aggregation
        if self._secagg_random is not None:

            if encryption_factors is None:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}: Secure aggregation random validation has been set but the encryption "
                    f"factors are not provided. Please provide encrypted `secagg_random` values in different parties. "
                    f"Or to not set/get `secagg_random()` before the aggregation.")

            logger.info("Validating secure aggregation results...")
            encryption_factors = [f for k, f in encryption_factors.items()]
            validation: List[float] = aggregate(params=encryption_factors)

            if len(validation) != 1 or not math.isclose(validation[0], self._secagg_random, abs_tol=0.03):
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}: Aggregation is failed due to incorrect decryption."
                )
            logger.info("Validation is completed.")

        elif encryption_factors is not None:
            logger.warning("Encryption factors are provided while secagg random is None. Please make sure secure "
                           "aggregation steps are applied correctly.")

        logger.info("Aggregating encrypted parameters. This process may take some time depending on model size.")
        # Aggregate parameters
        params = [p for _, p in model_params.items()]
        aggregated_params = aggregate(params=params)

        return aggregated_params

    def save_state(self) -> Dict[str, Any]:
        """Saves state of the secagg

        Returns:
            The secagg state to be saved
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                'active': self._active,
                'timeout': self.timeout,
                'clipping_range': self.clipping_range,
            },
            "attributes": {
                "_biprime": self._biprime.save_state() if self._biprime is not None else None,
                "_servkey": self._servkey.save_state() if self._servkey is not None else None,
                "_job_id": self._job_id,
                "_parties": self._parties
            }
        }

        return state

    @classmethod
    def load_state(
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

        if state["attributes"]["_biprime"] is not None:
            state["attributes"]["_biprime"] = SecaggBiprimeContext. \
                load_state(state=state["attributes"]["_biprime"])

        if state["attributes"]["_servkey"] is not None:
            state["attributes"]["_servkey"] = SecaggServkeyContext. \
                load_state(state=state["attributes"]["_servkey"])

        # Set attributes
        for name, val in state["attributes"].items():
            setattr(secagg, name, val)

        return secagg
