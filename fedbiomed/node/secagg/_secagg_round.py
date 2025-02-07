"""
This file is originally part of Fed-BioMed
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from fedbiomed.common.constants import ErrorNumbers, SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.utils import (
    matching_parties_servkey,
    matching_parties_dh
)
from fedbiomed.common.secagg import SecaggCrypter, SecaggLomCrypter
from fedbiomed.node.secagg_manager import SecaggServkeyManager, SecaggDhManager


class _SecaggSchemeRound(ABC):
    """Common class for all secure aggregation types handling of a training round on node
    """

    _min_num_parties: int = 2
    """Min number of parties"""

    def __init__(
        self,
        node_id: str,
        secagg_arguments: Dict,
        experiment_id: str
    ) -> None:
        """Constructor of the class

        Args:
            node_id: Id of the active node.
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: Experiment identifier that secure aggregation round
                will be performed for.
        """


        secagg_clipping_range = secagg_arguments.get('secagg_clipping_range')
        if secagg_clipping_range is not None and not isinstance(secagg_clipping_range, int):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad secagg clipping range type in train "
                f"request: {type(secagg_clipping_range)}"
            )

        self._node_id = node_id
        self._secagg_clipping_range = secagg_clipping_range
        self._parties = secagg_arguments.get('parties', [])
        self._secagg_arguments = secagg_arguments
        self._secagg_random = secagg_arguments.get('secagg_random')
        self._experiment_id = experiment_id

        if len(self._parties) < self._min_num_parties:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad parties list in train request: {self._parties}")


    @property
    def secagg_random(self) -> float | None:
        """Checks which secagg random nounce is used.

        Returns:
            Secagg random
        """
        return self._secagg_random

    @abstractmethod
    def encrypt(
        self,
        params: List[float],
        current_round: int,
        weight: Optional[int] = None
    ) -> List[int]:
        """Encrypts model parameters after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """


class _JLSRound(_SecaggSchemeRound):
    """Class for Joye Libert secure aggregation handling of a training round on node"""

    _min_num_round: int = 3

    def __init__(
        self,
        db: str,
        node_id: str,
        secagg_arguments: Dict,
        experiment_id: str
    ) -> None:
        """Constructor of the class

        Args:
            db: Path to database file.
            node_id: ID of the active node.
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: unique ID of experiment

        Raises:
            FedbiomedSecureAggregationError: Invalid secure aggregation setup
        """
        super().__init__(node_id, secagg_arguments, experiment_id)

        # setup
        secagg_servkey_id = secagg_arguments.get('secagg_servkey_id')
        self._secagg_manager = SecaggServkeyManager(db)
        self._secagg_servkey = self._secagg_manager.get(
            secagg_id=secagg_servkey_id, experiment_id=self._experiment_id)

        if self._secagg_servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Server-key/user-key share for "
                f"secagg: {secagg_servkey_id} is not existing. Aborting train request."
            )

        if not matching_parties_servkey(self._secagg_servkey, self._parties):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Parties for this training don't match "
                f"parties of secagg servkey context {secagg_servkey_id}"
            )

    def encrypt(
        self,
        params: List[float],
        current_round: int,
        weight: Optional[int] = None
    ) -> List[int]:
        """Encrypts model parameters with Joye-Libert after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """
        return SecaggCrypter().encrypt(
            num_nodes=len(self._secagg_servkey["parties"]),  # -1: don't count researcher
            current_round=current_round,
            params=params,
            key=self._secagg_servkey["context"]["server_key"],
            biprime=self._secagg_servkey["context"]["biprime"],
            clipping_range=self._secagg_clipping_range,
            weight=weight,
        )


class _LomRound(_SecaggSchemeRound):
    """Class for LOM secure aggregation handling of a training round on node
    """
    _min_num_parties: int = 2

    def __init__(self, db, node_id, secagg_arguments: Dict, experiment_id: str):
        """Constructor of the class

        Args:
            db: Path to database file.
            node_id: ID of the active node.
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: unique ID of experiment

        Raises:
            FedbiomedSecureAggregationError: no matching secagg context for this ID
            FedbiomedSecureAggregationError: parties in secagg context don't match experiment
        """
        super().__init__(node_id, secagg_arguments, experiment_id)

        secagg_dh_id = secagg_arguments.get('secagg_dh_id')
        self._secagg_manager = SecaggDhManager(db)
        self._secagg_id = secagg_dh_id
        self._secagg_dh = self._secagg_manager.get(secagg_id=secagg_dh_id, experiment_id=experiment_id)

        if self._secagg_dh is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Diffie Hellman context for "
                f"secagg: {secagg_dh_id} is not existing. Aborting train request."
            )

        if not matching_parties_dh(self._secagg_dh, self._parties):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Parties for this training don't match "
                f"parties of secagg Diffie Hellman context {secagg_dh_id}"
            )

        self.crypter = SecaggLomCrypter(nonce=self._secagg_id)


    def encrypt(
        self,
        params: List[float],
        current_round: int,
        weight: Optional[int] = None
    ) -> List[int]:
        """Encrypts model parameters with LOM after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """
        return self.crypter.encrypt(
            node_ids=self._secagg_dh["parties"],  # -1: don't count researcher
            node_id=self._node_id,
            current_round=current_round,
            params=params,
            pairwise_secrets=self._secagg_dh['context'],
            clipping_range=self._secagg_clipping_range,
            weight=weight,
        )


class SecaggRound:  # pylint: disable=too-few-public-methods
    """This class wraps secure aggregation schemes

    Attributes:
        scheme: Secure aggregation scheme
        use_secagg: True if secure aggregation is activated for round
    """

    element2class = {
        SecureAggregationSchemes.JOYE_LIBERT.value: _JLSRound,
        SecureAggregationSchemes.LOM.value: _LomRound
    }

    def __init__(
        self,
        db: str,
        node_id: str,
        secagg_arguments: Dict[str, Any] | None,
        secagg_active: bool,
        force_secagg: bool,
        experiment_id: str
    ) -> None:
        """Constructor of the class"""

        self._node_id = node_id
        self._secagg_active = secagg_active
        self._force_secagg = force_secagg

        self.use_secagg: bool = False
        self.scheme: _SecaggSchemeRound | None = None

        if not secagg_arguments and self._force_secagg:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Node requires to apply secure aggregation but "
                f"training request does not define it.")

        if secagg_arguments:
            if not self._secagg_active:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB318.value} Requesting secure aggregation while "
                    "it's not activated on the node."
                )

            sn = secagg_arguments.get('secagg_scheme')

            if sn is None:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB318.value}: Secagg scheme value missing in "
                    "the argument `secagg_arguments`"
                )
            try:
                _scheme = SecureAggregationSchemes(sn)
            except ValueError as e:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB318.value}: Bad secagg scheme value in train request: {sn}"
                ) from e

            self.scheme = SecaggRound.element2class[_scheme.value](
                db, node_id, secagg_arguments, experiment_id
            )
            self.use_secagg = True
