import functools
import math
import random
from typing import List, Union, Dict, Any

from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.secagg import SecaggCrypter
from fedbiomed.common.logger import logger


class SecureAggregation:
    parties: List[str]
    experiment_id: Union[None, str]

    _servkey: Union[SecaggServkeyContext, None]
    _biprime: Union[SecaggBiprimeContext, None]
    _secagg_crypter: SecaggCrypter

    def __init__(self,
                 active: bool = False,
                 timeout: int = 10,
                 clipping_range: Union[None, int] = None,
                 experiment_id: Union[None, str] = None
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
                parameters on the node side The default will be
                [`VEParameters.CLIPPING_RANGE`][fedbiomed.common.constants.VEParameters].
                The default value will be automatically set on the node side.

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

        self.parties = []
        self.experiment_id = experiment_id
        self.active = active
        self.timeout = timeout
        self.clipping_range = clipping_range

        self._servkey = None
        self._biprime = None
        self._clipping_range = None
        self._secagg_random = None
        self._secagg_crypter = SecaggCrypter()

    def activate(self, status):
        """Set activate status of secure aggregation

        Returns:
            Status of secure aggregation True if it is activated
        """

        if not isinstance(status, bool):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: The argument `status` for activation should be True or False, "
                f"but got {type(status)} "
            )

        self.active = status

        return self.active

    def secagg_random(self) -> float:
        """Assigns and returns random float to validate secure aggregation

        Returns:
            A random value for validation purposes
        """

        self._secagg_random = round(random.uniform(0, 1), 3)

        return self._secagg_random

    def secagg_biprime_id(self) -> str:
        """Gets secure aggregation server-key element id from `SecaggServkeyContext`


        Returns:
            Server-key context id
        """
        return self._servkey.secagg_id if self._servkey is not None else None

    def secagg_servkey_id(self):
        """Gets secure aggregation Biprime element id from `SecaggBiprimeContext`


        Returns:
            Biprime context id
        """
        return self._biprime.secagg_id if self._biprime is not None else None

    def setup(self):
        """Setup secure aggregation instruments

        Returns:
            Status of setup
        """

        if not self._biprime.status:
            self._biprime.setup(timeout=self.timeout)

        if not self._servkey.status:
            self._servkey.setup(timeout=self.timeout)

        return True

    def _set_secagg_contexts(self, parties: List[str]) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `experiment_id` and `parties` are set

        Args:
            parties: Parties that participates secure aggregation

        """

        self.parties = parties

        # TODO: support other options than using `default_biprime0`
        self._biprime = SecaggBiprimeContext(
            parties=self.parties,
            secagg_id='default_biprime0'
        )

        self._servkey = SecaggServkeyContext(
            parties=self.parties,
            job_id=self.experiment_id
        )

    def configure_round(
            self,
            parties: List[str],
            experiment_id: Union[None, str] = None
    ) -> None:
        """Configures secure aggregation for each round.

        This method checks the round state and creates secagg context element if
        not existing or re-instantiates if the state of the round has changes in cases of
        adding new nodes to the FL training

        Args:
            parties: Nodes that participates federated training
            experiment_id: The id of the experiment (Currently associated with Job id)

        Raises
            FedbiomedSecureAggregationError: Inconsistent experiment ID compare to
                previous rounds of the experiments.
        """
        # Experiment id can be also set thorough this method since on Experiment level `SecureAggregation` intance
        # is instantiated before instantiating the job class that hold experiment id
        # FIXME: Use only `__init__` `experiment_id` argument once experiment class creates and id instead of
        #  using `Job.id`
        if self.experiment_id is None and experiment_id is None:
            raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}There is no experiment (job) id associated to the "
                    f"secure aggregation. Please provide and experiment id using the argument `experiment_id`"
            )

        # Sets given experiment id by validating previous experiment id
        if experiment_id is not None:
            if self.experiment_id is not None and self.experiment_id != experiment_id:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}Experiment id of the secure aggregation can not "
                    f"be change in the middle of an experiment. Please create new experiment."
                )
            else:
                self.experiment_id = experiment_id

        if not self.parties:
            self._set_secagg_contexts(parties)

        elif set(self.parties) != set(parties):
            print("Here")
            logger.info(f"Parties of the experiment has changed. Re-creating secure "
                        f"aggregation context creation for the experiment {self.experiment_id}")
            self._set_secagg_contexts(parties)

    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            model_params: Dict[str, List[int]],
            encryption_factors: Union[Dict[str, List[int]], None] = None,
    ) -> List[float]:
        """Aggregates given model parameters

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

        logger.info("Securely aggregating model parameters...")

        aggregate = functools.partial(self._secagg_crypter.aggregate,
                                      current_round=round_,
                                      num_nodes=num_nodes,
                                      key=key,
                                      total_sample_size=total_sample_size,
                                      biprime=biprime)

        # Validate secure aggregation
        if self._secagg_random is not None:

            if encryption_factors is None:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}: Secure aggregation random validation has been set but the encryption "
                    f"factors are not provided. Please provide encrypted `secagg_random` values in different parties. "
                    f"Or to not set/get `secagg_random()` before the aggregation.")

            encryption_factors = [f for k, f in encryption_factors.items()]
            validation: List[float] = aggregate(params=encryption_factors)

            if len(validation) != 1 or not math.isclose(validation[0], self._secagg_random, abs_tol=0.01):
                raise FedbiomedSecureAggregationError(
                        f"{ErrorNumbers.FB417.value}: Aggregation is failed due to incorrect decryption."
                )

        elif encryption_factors is not None:
            logger.warning("Encryption factors are provided while secagg random is None. Please make sure secure "
                           "aggregation steps are applied correctly.")

        # Aggregate parameters
        params = [p for _, p in model_params.items()]
        aggregated_params = aggregate(params=params)

        return aggregated_params

    def save_state(self) -> Dict[str, Any]:
        """Saves stat of the secagg """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                'experiment_id': self.experiment_id,
                'active': self.active,
                'timeout': self.timeout,
                'clipping_range': self.clipping_range,
            },
            "attributes": {
                "_biprime": self._biprime.save_state() if self._biprime is not None else None,
                "_servkey": self._servkey.save_state() if self._servkey is not None else None,
                "parties": self.parties
            }
        }

        return state

    @classmethod
    def load_state(
            cls,
            state: Dict
    ) -> 'SecureAggregation':

        secagg = cls(**state["arguments"])
        secagg.parties = state["attributes"]["parties"]

        if state["attributes"]["_biprime"] is not None:
            state["attributes"]["_biprime"] = SecaggBiprimeContext.\
                load_state(state=state["attributes"]["_biprime"])

        if state["attributes"]["_servkey"] is not None:
            state["attributes"]["_servkey"] = SecaggServkeyContext.\
                load_state(state=state["attributes"]["_servkey"])

        # Set attributes
        for name, val in state["attributes"].items():
            setattr(secagg, name, val)

        return secagg
