# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Literal, Callable, Dict, List, Optional, Union, cast

from fedbiomed.common.constants import ErrorNumbers, SecureAggregationSchemes
from fedbiomed.common.exceptions import FedbiomedSecureAggregationError
from fedbiomed.common.logger import logger
from fedbiomed.common.secagg import SecaggCrypter, SecaggLomCrypter

from ._secagg_context import SecaggDHContext, SecaggServkeyContext


class SecureAggregation:
    """Interface for different secure aggregation classes"""

    def __init__(
        self,
        *args,
        scheme: SecureAggregationSchemes = SecureAggregationSchemes.LOM,
        **kwargs,
    ) -> None:
        """Constructs secure aggregation class

        Builds corresponding secure aggregation object/scheme based
        on given scheme argument

        Args:
            scheme: Secure aggregation scheme
        """

        self.__scheme = scheme

        match self.__scheme:
            case SecureAggregationSchemes.LOM:
                self.__secagg = LomSecureAggregation(*args, **kwargs)
            case SecureAggregationSchemes.JOYE_LIBERT:
                self.__secagg = JoyeLibertSecureAggregation(*args, **kwargs)
            case _:
                self.__secagg = LomSecureAggregation(*args, **kwargs)

    def __getattr__(self, item: str):
        """Wraps all functions/attributes of class members.

        Args:
             item: Requested item from class
        """

        if item in ("save_state_breakpoint"):
            return object.__getattribute__(self, item)

        return self.__secagg.__getattribute__(item)

    @abstractmethod
    def save_state_breakpoint(self) -> Dict[str, Any]:
        """Saves state of the secagg

        This methods also save states of `__secagg` which provides
        a single entry point for secagg schemes

        Returns:
            The secagg state to be saved
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                "scheme": self.__scheme.value,
            },
            "attributes": {},
            "attributes_states": {
                "_SecureAggregation__secagg": self.__secagg.save_state_breakpoint()
            },
        }

        return state

    @classmethod
    def load_state_breakpoint(cls, state: Dict) -> "SecureAggregation":
        """Create a `SecureAggregation` object from a saved state

        Args:
            state: saved state to restore in the created object

        Returns:
            The created `SecureAggregation` object
        """
        secagg = cls(scheme=SecureAggregationSchemes(state["arguments"]["scheme"]))

        for name, val in state["attributes_states"].items():

            _sub_cls = getattr(importlib.import_module(val["module"]), val["class"])
            instance = _sub_cls.load_state_breakpoint(val)
            setattr(secagg, name, instance)

        return secagg


class _SecureAggregation(ABC):
    """Secure aggregation controller of researcher component.

    This class is responsible for;

    - setting up the context for secure aggregation
    - Applying secure aggregation after receiving encrypted model parameters from nodes

    Attributes:
        clipping_range: Clipping range that will be used for quantization of model
            parameters on the node side.
        _parties: Nodes and researcher that participates federated training
        _experiment_id: ID of the current experiment.
        _secagg_crypter: Secure aggregation encrypter and decrypter to decrypt encrypted model
            parameters.
        _secagg_random: Random float generated tobe sent to node to validate secure aggregation
            after aggregation encrypted parameters, or None if validation is not used.
        _scheme: Secure aggregation scheme implemented by this class
    """

    @abstractmethod
    def __init__(
        self,
        active: bool = True,
        clipping_range: Union[None, int] = None,
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

        if clipping_range is not None and not isinstance(clipping_range, int):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Clipping range should be None or an integer, "
                f"but got not {type(clipping_range)}"
            )

        self.clipping_range: Optional[int] = clipping_range

        self._active: bool = active
        self._parties: Optional[List[str]] = None
        self._experiment_id: Optional[str] = None
        self._secagg_random: Optional[float] = None
        self._secagg_crypter: Union[SecaggCrypter, SecaggLomCrypter, None] = None
        self._scheme: SecureAggregationSchemes.LOM | Secure

    @property
    def parties(self) -> Union[List[str], None]:
        """Gets secagg parties

        Returns:
            List of secagg parties if it exists, or None
        """
        return self._parties

    @property
    def experiment_id(self) -> Union[str, None]:
        """Gets secagg associated experiment_id

        Returns:
            str of associated experiment_id if it exists, or None
        """
        return self._experiment_id

    @property
    def active(self) -> bool:
        """Gets secagg activation status

        Returns:
            bool, True if secagg is activated
        """
        return self._active

    @property
    def scheme(self) -> SecureAggregationSchemes | None:
        """Gets secagg scheme used

        Returns:
            Secagg scheme used
        """
        return self._scheme

    def activate(self, status) -> bool:
        """Set activate status of secure aggregation

        Returns:
            Status of secure aggregation True if it is activated
        """

        if not isinstance(status, bool):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: The argument `status` for activation "
                f"should be True or False, but got {type(status)} "
            )

        self._active = status

        return self._active

    @abstractmethod
    def train_arguments(self) -> Dict:
        """Gets train arguments for secagg train request

        Returns:
            Arguments that are going to be attached to the experiment.
        """
        return {
            "secagg_random": self._secagg_random,
            "secagg_clipping_range": self.clipping_range,
            "secagg_scheme": self._scheme.value,
            "parties": self._parties,
        }

    @abstractmethod
    def setup(
        self,
        parties: List[str],
        experiment_id: str,
        researcher_id: str,
        force: bool = False,
        insecure_validation: bool = True
    ) -> bool:

        """Setup secure aggregation instruments.

        Requires setting `parties` and `experiment_id` if they are not set in previous secagg
        setups. It is possible to execute without any argument if SecureAggregation
        has already `parties` and `experiment_id` defined. This feature provides researcher
        execute `secagg.setup()` if any connection issue

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
            researcher_id: ID of the researcher that context will be created for.
            force: Forces secagg setup even context is already existing
            insecure_validation: True if the insecure mechanism for validation secagg data
                coherence is enabled

        Raises
            FedbiomedSecureAggregationError: Invalid argument type
        """

        if not isinstance(parties, list):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `parties` list but "
                f"got {type(parties)}"
            )

        if not isinstance(experiment_id, str):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Expected argument `experiment_id` "
                f"string but got {type(parties)}"
            )

        self._configure_round(researcher_id, parties, experiment_id, insecure_validation)

    @abstractmethod
    def _set_secagg_contexts(self, researcher_id:str, parties: List[str], experiment_id: str) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `experiment_id` and `parties` are set

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
        """

        self._parties = parties

        # Updates experiment id if it is provided
        self._experiment_id = experiment_id

    def _configure_round(
        self,
        researcher_id: str,
        parties: List[str],
        experiment_id: str,
        insecure_validation: bool = True
    ) -> None:
        """Configures secure aggregation for each round.

        This method checks the round state and creates secagg context element if
        not existing or re-instantiates if the state of the round has changes in cases of
        adding new nodes to the FL training

        Args:
            parties: Nodes that participates federated training
            experiment_id: The id of the experiment
        """

        self._secagg_random = None
        if insecure_validation is True:
            # For each round it generates new secagg random float
            self._secagg_random = round(random.uniform(0, 1), 3)

        if self._parties is None or self._experiment_id != experiment_id:
            self._set_secagg_contexts(researcher_id, parties, experiment_id)

        elif set(self._parties) != set(parties):
            logger.info(
                f"Parties of the experiment has changed. Re-creating secure "
                f"aggregation context creation for the experiment {self._experiment_id}"
            )
            self._set_secagg_contexts(researcher_id, parties, experiment_id)

    @abstractmethod
    def aggregate(
        self, *args, model_params, total_sample_size, encryption_factors, **kwargs
    ) -> List[List[int]]:
        """Algorithm specific aggregation implementation"""

    def _validate(
        self,
        aggregate: functools.partial[List[int]],
        encryption_factors: Dict[str, Union[List[int], None]],
        num_expected_params: int | None = None,
    ) -> None:
        """Validate given inputs"""

        if any(v is None for v in encryption_factors.values()):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Secure aggregation consistency insecure "
                "validation has been set on the researcher but the encryption factors "
                "are not provided. Some nodes may use `FBM_SECURIRY_SECAGG_INSECURE_VALIDATION` to "
                "`False` for security reason. Please use consistent setup "
                "between researcher and nodes."
            )

        logger.info("Validating secure aggregation results...")
        encryption_factors = [f for k, f in encryption_factors.items()]

        validation: List[float]

        if num_expected_params:
            validation = aggregate(params=encryption_factors, num_expected_params=1)
        else:
            validation = aggregate(params=encryption_factors)
        if len(validation) != 1 or not math.isclose(
            validation[0], self._secagg_random, abs_tol=0.03
        ):
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Aggregation has failed due to incorrect decryption."
            )
        logger.info("Validation is completed.")

    def _aggregate(
        self,
        model_params: Dict[str, List[int]],
        encryption_factors: Dict[str, Union[List[int], None]],
        aggregate: functools.partial[List[int]],
        num_expected_params: int | None = None,
    ) -> List[float]:
        """Aggregates given model parameters

        Args:
            total_sample_size: sum of number of samples used by all nodes
            model_params: model parameters from the participating nodes
            encryption_factors: encryption factors from the participating nodes
            num_expected_params: number of decrypted parameters to decode from the model parameters.
                It is an optional parameter since some schemes does not require it.
            aggregate: partial function for aggregation

        Returns:
            Aggregated parameters

        Raises:
            FedbiomedSecureAggregationError: secure aggregation context not properly configured
            FedbiomedSecureAggregationError: secure aggregation computation error
        """
        if self._secagg_random is not None:
            self._validate(aggregate, encryption_factors, num_expected_params)

        logger.info(
            "Aggregating encrypted parameters. This process may take some time depending"
            "on model size."
        )
        # Aggregate parameters
        # if isinstance(model_params, dict):
        params = list(model_params.values())  # convert dict into list of list
        # from now forward params is of type List[List[int]]
        # else:
        #     params = model_params

        if num_expected_params:
            aggregated_params = aggregate(
                params=params, num_expected_params=num_expected_params
            )
        else:
            aggregated_params = aggregate(params=params)

        return aggregated_params

    @abstractmethod
    def save_state_breakpoint(self) -> Dict[str, Any]:
        """Saves state of the secagg

        Returns:
            The secagg state to be saved
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "arguments": {
                "active": self._active,
                "clipping_range": self.clipping_range,
            },
            "attributes": {
                "_experiment_id": self._experiment_id,
                "_parties": self._parties,
            },
        }

        return state

    @classmethod
    @abstractmethod
    def load_state_breakpoint(cls, state: Dict) -> "SecureAggregation":
        """Create a `SecureAggregation` object from a saved state

        Args:
            state: saved state to restore in the created object

        Returns:
            The created `SecureAggregation` object
        """
        secagg = cls(**state["arguments"])

        # Set attributes
        for name, val in state["attributes"].items():
            setattr(secagg, name, val)

        return secagg


class JoyeLibertSecureAggregation(_SecureAggregation):
    """Secure aggregation controller of researcher component for Joye-Libert.

    This class is responsible for;

    - setting up the context for Joye-Libert secure aggregation
    - Applying secure aggregation after receiving encrypted model parameters from nodes

    Attributes:
        _servkey: Server-key context setup instance.
    """

    def __init__(
        self,
        active: bool = True,
        clipping_range: Union[None, int] = None,
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
        super().__init__(active, clipping_range)

        self._servkey: SecaggServkeyContext | None = None
        self._secagg_crypter: SecaggCrypter = SecaggCrypter()
        self._scheme = SecureAggregationSchemes.JOYE_LIBERT

    @property
    def servkey(self) -> Union[None, SecaggServkeyContext]:
        """Gets servkey object

        Returns:
            Servkey object, None if servkey is not setup
        """
        return self._servkey

    def train_arguments(self) -> Dict:
        """Gets train arguments for secagg train request

        Returns:
            Arguments that are going to be attached to the experiment.
        """
        arguments = super().train_arguments()
        arguments.update(
            {
                "secagg_servkey_id": (
                    self._servkey.secagg_id if self._servkey is not None else None
                ),
            }
        )
        return arguments

    def setup(
        self,
        parties: List[str],
        experiment_id: str,
        researcher_id: str,
        force: bool = False,
        insecure_validation: bool = True
    ) -> bool:
        """Setup secure aggregation instruments.

        Requires setting `parties` and `experiment_id` if they are not set in previous secagg
        setups. It is possible to execute without any argument if SecureAggregation
        has already `parties` and `experiment_id` defined. This feature provides researcher
        execute `secagg.setup()` if any connection issu#e

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
            researcher_id: ID of the researcher that context will be created for.
            force: Forces secagg setup even context is already existing
            insecure_validation: True if the insecure mechanism for validation secagg data
                coherence is enabled

        Returns:
            Status of setup

        Raises
            FedbiomedSecureAggregationError: Invalid argument type
        """
        super().setup(parties, experiment_id, researcher_id, force, insecure_validation)

        self._servkey = cast(SecaggServkeyContext, self._servkey)
        if not self._servkey.status or force:
            self._servkey.setup()

        return True

    def _set_secagg_contexts(
        self,
        researcher_id: str,
        parties: List[str],
        experiment_id: str
    ) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `experiment_id` and `parties` are set

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
        """
        super()._set_secagg_contexts(researcher_id, parties, experiment_id)

        self._servkey = SecaggServkeyContext(
            researcher_id=researcher_id, parties=self._parties, experiment_id=self._experiment_id
        )

    def aggregate(
        self,
        *args,
        round_: int,
        total_sample_size: int,
        model_params: Dict[str, List[int]],
        encryption_factors: Dict[str, Union[List[int], None]] = {},
        num_expected_params: int = 1,
        **kwargs,
    ) -> List[float]:
        """Aggregates given model parameters

        Args:
            round_: current training round number
            total_sample_size: sum of number of samples used by all nodes
            model_params: model parameters from the participating nodes
            encryption_factors: encryption factors from the participating nodes
            num_expected_params: number of decrypted parameters to decode from the model parameters

        Returns:
            Aggregated parameters

        Raises:
            FedbiomedSecureAggregationError: secure aggregation context not properly configured
        """

        if self._servkey is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, Servkey context is "
                f"not configured. Please setup secure aggregation before the aggregation."
            )

        if not self._servkey.status:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, one of Biprime or Servkey context is "
                f"not set properly"
            )

        key = self._servkey.context["server_key"]
        biprime = self._servkey.context["biprime"]
        num_nodes = len(model_params)

        aggregate = functools.partial(
            self._secagg_crypter.aggregate,
            current_round=round_,
            num_nodes=num_nodes,
            key=key,
            total_sample_size=total_sample_size,
            biprime=biprime,
            clipping_range=self.clipping_range,
        )

        return self._aggregate(
            model_params,
            encryption_factors,
            aggregate,
            num_expected_params,
        )

    def save_state_breakpoint(self) -> Dict[str, Any]:
        """Saves state of the secagg

        Returns:
            The secagg state to be saved
        """
        state = super().save_state_breakpoint()

        state["attributes"].update(
            {
                "_servkey": (
                    self._servkey.save_state_breakpoint()
                    if self._servkey is not None
                    else None
                ),
            }
        )

        return state

    @classmethod
    def load_state_breakpoint(cls, state: Dict) -> "SecureAggregation":
        """Create a `SecureAggregation` object from a saved state

        Args:
            state: saved state to restore in the created object

        Returns:
            The created `SecureAggregation` object
        """
        if state["attributes"]["_servkey"] is not None:
            state["attributes"]["_servkey"] = (
                SecaggServkeyContext.load_state_breakpoint(
                    state=state["attributes"]["_servkey"]
                )
            )

        return super().load_state_breakpoint(state)


class LomSecureAggregation(_SecureAggregation):
    """Secure aggregation controller of researcher component for Low Overhead Masking.

    This class is responsible for;

    - setting up the context for LOM secure aggregation
    - Applying secure aggregation after receiving encrypted model parameters from nodes

    Attributes:
        _dh: Diffie Hellman keypairs per node context setup instance.
    """

    def __init__(
        self,
        active: bool = True,
        clipping_range: Union[None, int] = None,
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
        super().__init__(active, clipping_range)

        self._dh: SecaggDHContext | None = None
        self._secagg_crypter = SecaggLomCrypter()
        self._scheme = SecureAggregationSchemes.LOM


    @property
    def dh(self) -> Union[None, SecaggDHContext]:
        """Gets Diffie Hellman keypairs object

        Returns:
            DH object, None if DH is not setup
        """
        return self._dh

    def train_arguments(self) -> Dict:
        """Gets train arguments for secagg train request

        Returns:
            Arguments that are going to be attached to the experiment.
        """
        arguments = super().train_arguments()
        arguments.update(
            {
                "secagg_dh_id": self._dh.secagg_id if self._dh is not None else None,
            }
        )
        return arguments

    def setup(
        self,
        parties: List[str],
        experiment_id: str,
        researcher_id: str,
        force: bool = False,
        insecure_validation: bool = True
    ) -> bool:
        """Setup secure aggregation instruments.

        Requires setting `parties` and `experiment_id` if they are not set in previous secagg
        setups. It is possible to execute without any argument if SecureAggregation
        has already `parties` and `experiment_id` defined. This feature provides researcher
        execute `secagg.setup()` if any connection issue

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
            researcher_id: ID of the researcher that executes secagg setup.
            force: Forces secagg setup even if context is already existing
            insecure_validation: True if the insecure mechanism for validation secagg data
                coherence is enabled

        Returns:
            Status of setup

        Raises
            FedbiomedSecureAggregationError: Invalid argument type
        """

        parties = list(filter(lambda x: x != researcher_id, parties))

        super().setup(parties, experiment_id, researcher_id, force, insecure_validation)

        self._dh = cast(SecaggDHContext, self._dh)

        if not self._dh.status or force:
            self._dh.setup()

        return self._dh.status


    def _set_secagg_contexts(
        self,
        researcher_id: str,
        parties: List[str],
        experiment_id: str
    ) -> None:
        """Creates secure aggregation context classes.

        This function should be called after `experiment_id` and `parties` are set

        Args:
            parties: Parties that participates secure aggregation
            experiment_id: The id of the experiment
        """
        super()._set_secagg_contexts(researcher_id, parties, experiment_id)

        self._dh = SecaggDHContext(
            researcher_id=researcher_id, parties=self._parties, experiment_id=self._experiment_id
        )

    def aggregate(
        self,
        *args,
        model_params: Dict[str, List[int]],
        total_sample_size: int,
        encryption_factors: Dict[str, Union[List[int], None]] = {},
        **kwargs,
    ) -> List[float]:
        """Aggregates given model parameters

        Args:
            total_sample_size: sum of number of samples used by all nodes
            model_params: model parameters from the participating nodes
            encryption_factors: encryption factors from the participating nodes

        Returns:
            Aggregated parameters

        Raises:
            FedbiomedSecureAggregationError: secure aggregation context not properly configured
        """

        if self._dh is None:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, Diffie "
                "Hellman context is not configured. Please setup secure aggregation "
                "before the aggregation."
            )

        if not self._dh.status:
            raise FedbiomedSecureAggregationError(
                f"{ErrorNumbers.FB417.value}: Can not aggregate parameters, Diffie "
                "Hellman context is not set properly"
            )

        aggregate = functools.partial(
            self._secagg_crypter.aggregate,
            total_sample_size=total_sample_size,
            clipping_range=self.clipping_range,
        )

        return self._aggregate(
            model_params,
            encryption_factors,
            aggregate,
        )

    def save_state_breakpoint(self) -> Dict[str, Any]:
        """Saves state of the secagg

        Returns:
            The secagg state to be saved
        """
        state = super().save_state_breakpoint()

        state["attributes"].update(
            {
                "_dh": (
                    self._dh.save_state_breakpoint() if self._dh is not None else None
                ),
            }
        )

        return state

    @classmethod
    def load_state_breakpoint(cls, state: Dict) -> "SecureAggregation":
        """Create a `SecureAggregation` object from a saved state

        Args:
            state: saved state to restore in the created object

        Returns:
            The created `SecureAggregation` object
        """
        if state["attributes"]["_dh"] is not None:
            state["attributes"]["_dh"] = SecaggDHContext.load_state_breakpoint(
                state=state["attributes"]["_dh"]
            )

        return super().load_state_breakpoint(state)
