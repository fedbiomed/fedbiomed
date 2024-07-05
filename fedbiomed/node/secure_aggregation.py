# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Union, Tuple, List, Optional
from abc import ABC, abstractmethod

from fedbiomed.common.constants import ErrorNumbers, SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.utils import matching_parties_servkey, matching_parties_biprime, \
    matching_parties_dh

from fedbiomed.node.environ import environ
from fedbiomed.common.secagg import SecaggCrypter, SecaggLomCrypter
from fedbiomed.node.secagg_manager import SKManager, BPrimeManager, DHManager


class BaseSecureAggregation(ABC):
    """Common class for all secure aggregation types handling of a training round on node
    """
    def __init__(self, scheme: SecureAggregationSchemes, secagg_arguments: Dict):
        """Constructor of the class

        Args:
            scheme: type of secure aggregation used
            secagg_arguments:  secure aggregation arguments from train request
        """
        self._scheme = scheme
        self._secagg_arguments = secagg_arguments

        self._use_secagg = False
        self._secagg_random = 0     # default value to protect call to getter

    def _check_secagg_args(self, component_ids: Tuple[str]) -> None:
        """Checks secure aggregations arguments for all cases where secagg is activated.

        Args:
            component_ids: IDs of secure aggregation components for the used type of secagg

        Raises:
            FedbiomedSecureAggregationError: bad secagg argument
        """
        if not environ["SECURE_AGGREGATION"]:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value} Requesting secure aggregation while it's not activated on the node."
            )

        secagg_random = self._secagg_arguments.get('secagg_random')
        if not isinstance(secagg_random, float):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad secagg random type in train request: {type(secagg_random)}"
            )
        self._secagg_random = secagg_random

        secagg_clipping_range = self._secagg_arguments.get('secagg_clipping_range')
        if secagg_clipping_range is not None and not isinstance(secagg_clipping_range, int):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad secagg clipping range type in train request: {type(secagg_random)}"
            )
        self._secagg_clipping_range = secagg_clipping_range

        parties = self._secagg_arguments.get('parties')
        if self._scheme is not SecureAggregationSchemes.NONE.value and \
                (not isinstance(parties, list) or len(parties) < 3 or not all([isinstance(p, str) for p in parties])):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad parties list in train request: {parties}")
        self._parties = parties

        # check ids are non empty strings, as it was not tested before
        for c_id in component_ids:
            if not isinstance(c_id, str) or not c_id:
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB318.value}: Bad secagg element: value '{c_id}', type {type(c_id)}"
                )

        self._use_secagg = True

    @property
    def use_secagg(self) -> bool:
        """Checks whether secagg is used or not.

        Returns:
            bool is True when secagg is used, False when not used
        """
        return self._use_secagg

    @property
    def scheme(self) -> SecureAggregationSchemes:
        """Checks which secagg scheme is used.

        Returns:
            Secure aggregation scheme
        """
        return self._scheme

    @property
    def secagg_random(self) -> float:
        """Checks which secagg random nounce is used.

        Returns:
            Secagg random nounce
        """
        return self._secagg_random

    @abstractmethod
    def encrypt(self, params: List[float], current_round: int, weight: Optional[int] = None) -> List[int]:
        """Encrypts model parameters after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """


class NoneSecureAggregation(BaseSecureAggregation):
    """Class to use when no secure aggregation during training round on node
    """
    def __init__(self, scheme: SecureAggregationSchemes, secagg_arguments: Dict, experiment_id: str):
        """Constructor of the class

        Args:
            scheme: type of secure aggregation used
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: unused argument

        Raises:
            FedbiomedSecureAggregationError: node requires secure aggregation
        """
        super().__init__(scheme, secagg_arguments)

        if environ["FORCE_SECURE_AGGREGATION"]:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Node requires to apply secure aggregation but "
                f"training request does not define it."
            )

    def encrypt(self, params: List[float], current_round: int, weight: Optional[int] = None) -> None:
        """Encrypts model parameters after local training.

        Args:
            params: unused
            current_round: unused
            weight: unused

        Raises:
            FedbiomedSecureAggregationError: always raise error, this method should never be called
        """
        raise FedbiomedSecureAggregationError(
            f"{ErrorNumbers.FB318.value}: Cannot encrypt parameters when secure aggregation is not active"
        )


class JoyeLibertSecureAggregation(BaseSecureAggregation):
    """Class for Joye Libert secure aggregation handling of a training round on node
    """
    def __init__(self, scheme: SecureAggregationSchemes, secagg_arguments: Dict, experiment_id: str):
        """Constructor of the class

        Args:
            scheme: type of secure aggregation used
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: unique ID of experiment

        Raises:
            FedbiomedSecureAggregationError: xxx
        """
        super().__init__(scheme, secagg_arguments)

        component_ids = (
            secagg_arguments.get('secagg_servkey_id'),
            secagg_arguments.get('secagg_biprime_id'),
        )
        self._check_secagg_args(component_ids)

        # setup
        secagg_servkey_id = secagg_arguments.get('secagg_servkey_id')
        secagg_biprime_id = secagg_arguments.get('secagg_biprime_id')
        self._secagg_biprime = BPrimeManager.get(secagg_id=secagg_biprime_id)
        self._secagg_servkey = SKManager.get(secagg_id=secagg_servkey_id, experiment_id=experiment_id)

        if self._secagg_biprime is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Biprime for secagg: {secagg_biprime_id} "
                f"is not existing. Aborting train request."
            )
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
        if not matching_parties_biprime(self._secagg_biprime, self._parties):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Parties for this training don't match "
                f"parties of secagg biprime context {secagg_biprime_id}"
            )

    def encrypt(self, params: List[float], current_round: int, weight: Optional[int] = None) -> List[int]:
        """Encrypts model parameters with Joye-Libert after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """
        return SecaggCrypter().encrypt(
            num_nodes=len(self._secagg_servkey["parties"]) - 1,  # -1: don't count researcher
            current_round=current_round,
            params=params,
            key=self._secagg_servkey["context"]["server_key"],
            biprime=self._secagg_biprime["context"]["biprime"],
            clipping_range=self._secagg_clipping_range,
            weight=weight,
        )


class LomSecureAggregation(BaseSecureAggregation):
    """Class for LOM secure aggregation handling of a training round on node
    """
    def __init__(self, scheme: SecureAggregationSchemes, secagg_arguments: Dict, experiment_id: str):
        """Constructor of the class

        Args:
            scheme: type of secure aggregation used
            secagg_arguments:  secure aggregation arguments from train request
            experiment_id: unique ID of experiment

        Raises:
            FedbiomedSecureAggregationError: no matching secagg context for this ID
            FedbiomedSecureAggregationError: parties in secagg context don't match experiment
        """
        super().__init__(scheme, secagg_arguments)

        component_ids = (
            secagg_arguments.get('secagg_dh_id'),
        )
        self._check_secagg_args(component_ids)

        # setup
        secagg_dh_id = secagg_arguments.get('secagg_dh_id')
        self._secagg_dh = DHManager.get(secagg_id=secagg_dh_id, experiment_id=experiment_id)

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

    def encrypt(self, params: List[float], current_round: int, weight: Optional[int] = None) -> List[int]:
        """Encrypts model parameters with LOM after local training.

        Args:
            params: list of flattened parameters
            current_round: current round of federated training
            weight: weight for the params

        Returns:
            List of encrypted parameters
        """
        return SecaggLomCrypter().encrypt(
            num_nodes=len(self._secagg_dh["parties"]) - 1,  # -1: don't count researcher
            current_round=current_round,
            params=params,
            temporary_key=self._secagg_dh['context'],
            clipping_range=self._secagg_clipping_range,
            weight=weight,
        )


class SecureAggregation:
    """Factory class for instantiating any type of node secure aggregation object
    """

    element2class = {
        SecureAggregationSchemes.NONE.value: NoneSecureAggregation,
        SecureAggregationSchemes.JOYE_LIBERT.value: JoyeLibertSecureAggregation,
        SecureAggregationSchemes.LOM.value: LomSecureAggregation
    }

    def __init__(self):
        """Constructor of the class
        """
        pass

    def __call__(self, secagg_arguments: Union[Dict, None], experiment_id: str) -> BaseSecureAggregation:
        """Instantiate a node secure aggregation object.

        Args:
            secagg_arguments: secure aggregation arguments from train request
            experiment_id: unique ID of experiment

        Returns:
            a new secure aggregation object
        """
        secagg_arguments = secagg_arguments if isinstance(secagg_arguments, dict) else {}
        secagg_scheme = secagg_arguments.get('secagg_scheme', SecureAggregationSchemes.NONE.value)
        try:
            scheme = SecureAggregationSchemes(secagg_scheme)
        except ValueError:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Bad secagg scheme value in train request: {secagg_scheme}"
            )

        try:
            return SecureAggregation.element2class[scheme.value](scheme, secagg_arguments, experiment_id)
        except Exception as e:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB318.value}: Can not instantiate secure aggregation objects. Error: {e}"
            )
