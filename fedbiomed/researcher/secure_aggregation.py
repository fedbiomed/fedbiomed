
from typing import List

from fedbiomed.researcher.secagg import SecaggServkeyContext, SecaggBiprimeContext
from fedbiomed.common.secagg import SecaggCrypter


class SecureAggregation:

    secagg_crypter = SecaggCrypter()

    def __init__(self, parties: List[str], experiment_id: str):
        """

        Args:
            parties: ...
            experiment_id: ...

        """
        self._parties = None
        self._servkey = None
        self._biprime = None
        self._experiment_id = None

    def setup(self, parties):
        """Setup secure aggregation instruments

        Args:
            ....

        Returns:
            ....
        """

        # Validate parties
        if set(self._parties) != set(parties):
            raise "Opps"

    def configure_round(
            self,
            parties: List[str],
            experiment_id: str,
            secagg_biprime_id: str,
            secagg_servkey_id: str
    ):

        if self._parties is None:
            self._parties = parties
            self._experiment_id = experiment_id
            self._servkey = SecaggServkeyContext(parties=parties, job_id=experiment_id)

            # TODO: support other options than using `default_biprime0`
            self._biprime = SecaggBiprimeContext(parties=parties, secagg_id='default_biprime0')


    def _configure_bi_prime(self, parties):

        self._biprime = SecaggBiprimeContext(parties=parties, secagg_id='default_biprime0')

    def _compare_with_previous_round(self):
        """Compares secure aggregation state with the previous round state.

        Returns:
            ....
        """



    @staticmethod
    def aggregate(model_params: List[List[int]]):
        """Aggregates given model parameters

        """
        pass
