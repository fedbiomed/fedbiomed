
import functools
import math
import random
from typing import List, Union, Dict

from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext
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
                 timeout: int = 10
    ) -> None:
        """Class constructor

        Assign default values for attributes
        """
        self.parties = []
        self.experiment_id = None
        self.active = active
        self.timeout = timeout

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
        return self._servkey.secagg_id

    def secagg_servkey_id(self):
        """Gets secure aggregation Biprime element id from `SecaggBiprimeContext`


        Returns:
            Biprime context id
        """
        return self._biprime.secagg_id

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

    def _set_secagg_contexts(self) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `experiment_id` and `parties` are set

        """

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
            experiment_id: str
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
        # Make sure that secure aggregation for the round is for same experiment
        if experiment_id is None:
            self.experiment_id = experiment_id
        elif self.experiment_id != experiment_id:
            raise f"Experiment id of the secure aggregation can not be change in the " \
                  f"middle of an experiment. Please create new experiment."

        if not self.parties:
            self.parties = parties
            self._set_secagg_contexts()

        elif set(self.parties) != set(parties):
            logger.info(f"Parties of the experiment has changed. Re-creating secure "
                        f"aggregation context creation for the experiment {self.experiment_id}")
            self._set_secagg_contexts()

    def aggregate(
            self,
            round_: int,
            total_sample_size: int,
            encryption_factors: List[Dict[str, List[int]]],
            model_params: List[List[int]]
    ) -> List[float]:
        """Aggregates given model parameters

        """

        biprime = self._biprime.context["context"]["biprime"]
        key = self._biprime.context["context"]["server_key"]

        num_nodes = len(model_params)

        logger.info("Securely aggregating model parameters...")

        aggregate = functools.partial(self._secagg_crypter.aggregate,
                                      current_round=round_,
                                      num_nodes=num_nodes,
                                      key=key,
                                      total_sample_size=total_sample_size,
                                      biprime=biprime)

        # Validate secure aggregation
        encryption_factors = [f for k, f in encryption_factors.items()]
        validation: List[float] = aggregate(params=encryption_factors)

        if len(validation) != 1 or not math.isclose(validation[0], self._secagg_random, abs_tol=0.01):
            raise "Aggregation is failed due to incorrect decryption."

        # Aggregate parameters
        params = [p for _, p in model_params.items()]
        aggregated_params = aggregate(params=params)

        return aggregated_params

