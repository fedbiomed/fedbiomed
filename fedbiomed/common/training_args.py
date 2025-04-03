# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Provide a way to easily to manage training arguments.
"""
from copy import deepcopy
from typing import Any, Dict, Type, TypeVar, Union, Tuple, Callable

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedUserInputError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.validator import SchemeValidator, ValidatorError, \
    ValidateError, RuleError, validator_decorator


@validator_decorator
def _validate_dp_type(value: Any):
    """ Validates whether DP type is valid"""
    if value not in ["central", "local"]:
        return False, f"DP type should one of `central` or `local` not {value}"
    else:
        return True


DPArgsValidator = SchemeValidator({
    'type': {
        "rules": [str, _validate_dp_type], "required": True, "default": "central"
    },
    'sigma': {
        "rules": [float], "required": True
    },
    'clip': {
        "rules": [float], "required": True
    },
})


class TrainingArgs:
    """
    Provide a container to manage training arguments.

    This class uses the Validator and SchemeValidator classes
    and provides a default scheme, which describes the arguments
    necessary to train/validate a TrainingPlan.

    It also permits to extend the TrainingArgs then testing new features
    by supplying an extra_scheme at TrainingArgs instantiation.
    """

    def __init__(self, ta: Dict = None, extra_scheme: Dict = None, only_required: bool = True):
        """
        Create a TrainingArgs from a Dict with input validation.

        Args:
            ta:     dictionary describing the TrainingArgs scheme.
                    if empty dict or None, a minimal instance of TrainingArgs
                    will be initialized with default values for required keys
            extra_scheme: user provided scheme extension, which add new rules or
                    update the scheme of the default training args.
                    Warning: this is a dangerous feature, provided to
                    developers, to ease the test of future Fed-Biomed features
            only_required: if True, the object is initialized only with required
                    values defined in the default_scheme (+ extra_scheme).
                    If False, then all default values will also be returned
                    (not only the required key/value pairs).

        Raises:
            FedbiomedUserInputError: in case of bad value or bad extra_scheme
        """

        self._scheme = TrainingArgs.default_scheme()

        if not isinstance(extra_scheme, dict):
            extra_scheme = {}

        for k in extra_scheme:
            self._scheme[k] = extra_scheme[k]

        try:
            self._sc = SchemeValidator(self._scheme)
        except RuleError as e:
            #
            # internal error (invalid scheme)
            msg = ErrorNumbers.FB414.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

        # scheme is validated from here
        if ta is None:
            ta = {}

        try:
            self._ta = self._sc.populate_with_defaults(ta, only_required=only_required)
        except ValidatorError as e:
            # scheme has required keys without defined default value
            msg = ErrorNumbers.FB414.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

        try:
            self._sc.validate(self._ta)
        except ValidateError as e:
            # transform to a Fed-BioMed error
            msg = ErrorNumbers.FB414.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

        # Validate DP arguments if it is existing in training arguments
        if self._ta["dp_args"] is not None:
            try:
                self._ta["dp_args"] = DPArgsValidator.populate_with_defaults(self._ta["dp_args"], only_required=False)
                DPArgsValidator.validate(self._ta["dp_args"])
            except ValidateError as e:
                msg = f"{ErrorNumbers.FB414.value}: {e}"
                logger.critical(msg)
                raise FedbiomedUserInputError(msg)

    def testing_arguments(self) -> Dict:
        """ Extract testing arguments from training arguments

        Returns:
            Testing arguments as dictionary
        """
        keys = ['test_ratio', 'test_on_local_updates', 'test_on_global_updates',
                'test_metric', 'test_metric_args', 'test_batch_size', 'shuffle_testing_dataset']
        return self._extract_args(keys)

    def loader_arguments(self) -> Dict:
        """ Extracts data loader arguments

        Returns:
            The dictionary of arguments for dataloader
        """
        return self["loader_args"]

    def optimizer_arguments(self) -> Dict:

        return self["optimizer_args"]

    def pure_training_arguments(self):
        """ Extracts the arguments that are only necessary for training_routine

        Returns:
            Contains training argument for training routine
        """

        keys = ["batch_maxnum",
                "fedprox_mu",
                "log_interval",
                "share_persistent_buffers",
                "dry_run",
                "epochs",
                "use_gpu",
                "num_updates"]
        return self._extract_args(keys)

    def dp_arguments(self):
        """Extracts the arguments for differential privacy

        Returns:
            Contains differential privacy arguments
        """
        return self["dp_args"]

    def _extract_args(self, keys) -> Dict:
        """Extract arguments by given array of keys

        Returns:
            Contains key value peer of given keys
        """
        return {arg: self[arg] for arg in keys}

    @staticmethod
    def _nonnegative_integer_value_validator_hook(name: str) -> Callable:
        @validator_decorator
        def _named_nonnegative_integer_value_validator_hook(val: Union[int, None]) -> Union[Tuple[bool, str], bool]:
            if val is None or isinstance(val, (float, int)):
                if val is not None:
                    if int(val) != float(val) or val < 0:
                        return False, f"{name} should be a non-negative integer or None, but got {val}"
                return True
            else:
                return False, f"{name} should be a non-negative integer or None, but got {val}"
        return _named_nonnegative_integer_value_validator_hook

    @staticmethod
    @validator_decorator
    def _metric_validation_hook(metric: Union[MetricTypes, str, None]) -> Union[bool, str]:
        """
        Validate the metric argument of test_metric.
        """
        if metric is None:
            return True

        if isinstance(metric, MetricTypes):
            return True

        if isinstance(metric, str):
            metric = metric.upper()
            if metric in MetricTypes.get_all_metrics():
                return True

        return False, f"Metric {metric} is not a supported Metric"

    @staticmethod
    @validator_decorator
    def _fedprox_mu_validator(val: Union[float, None]) -> Union[Tuple[bool, str], bool]:
        """ Validates fedprox_mu value whether it None or float

        Returns:
            Validation status  or/and error message
        """
        if isinstance(val, float):
            return True
        elif val is None:
            return True
        return False, f"Expected `fedprox_mu` value is float, but got {type(val)}. "

    @staticmethod
    @validator_decorator
    def _test_ratio_hook(v: Any) -> bool:
        """
        Test if in [ 0.0 , 1.0]  interval.
        """
        if v < 0 or v > 1:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _lr_hook(v: Any):
        """
        Test if lr is greater than 0.
        """
        if v < 0:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _validate_dp_args(v: Any):
        """
        Test if lr is greater than 0.
        """
        if v is None:
            return True
        elif not isinstance(v, dict):
            return False, f"`dp_args` should be None or dictionary, not {type(v)}"

        return True

    @staticmethod
    def optional_type(typespec: Union[Type, Tuple[Type, ...]], argname: str):
        """Utility factory function to generate functions that check for an optional type(s).

        Args:
            typespec: type specification which will be passed to the `isinstance` function
            argname: the name of the training argument for outputting meaningful error messages

        Returns:
            type_check: a callable that takes a single argument and checks whether it is either None
                or the required type(s)
        """
        @validator_decorator
        def type_check(v):
            if v is not None and not isinstance(v, typespec):
                return False, f"Invalid type: {argname} must be {typespec} or None"
            return True
        return type_check

    @classmethod
    def default_scheme(cls) -> Dict:
        """
        Returns the default (base) scheme for TrainingArgs.

        A summary of the semantics of each argument is given below. Please refer to the source code of this function
        for additional information on typing and constraints.

        | argument | meaning |
        | -------- | ------- |
        | optimizer_args | supplemental arguments for initializing the optimizer |
        | loader_args | supplemental arguments passed to the data loader |
        | epochs | the number of epochs performed during local training on each node |
        | num_updates | the number of model updates performed during local training on each node. Supersedes epochs if both are specified |
        | use_gpu | toggle requesting the use of GPUs for local training on the node when available, propagates to `declearn's GPU |
        | dry_run | perform a single model update for testing on each node and correctly handle GPU execution |
        | batch_maxnum | prematurely break after batch_maxnum model updates for each epoch (useful for testing) |
        | test_ratio | the proportion of validation samples to total number of samples in the dataset |
        | test_batch_size | batch size used for testing trained model wrt a set of metric |
        | test_on_local_updates | toggles validation after local training |
        | test_on_global_updates | toggles validation before local training |
        | shuffle_testing_dataset | whether reset or not the testing (and training) dataset for this `Round` |
        | test_metric | metric to be used for validation |
        | test_metric_args | supplemental arguments for the validation metric |
        | log_interval | output a training logging entry every log_interval model updates |
        | fedprox_mu | set the value of mu and enable FedProx correction |
        | dp_args | arguments for Differential Privacy |
        | share_persistent_buffers | toggle whether nodes share the full state_dict (when True) or only trainable parameters (False) in a TorchTrainingPlan |
        | random_seed | set random seed at the beginning of each round |

        """
        return {
            "optimizer_args": {
                "rules": [dict], "required": True, "default": {}
            },
            "loader_args": {
                "rules": [dict], "required": True, "default": {}
            },
            "epochs": {
                "rules": [cls._nonnegative_integer_value_validator_hook('epochs')], "required": True, "default": None
            },
            "num_updates": {
                "rules": [cls._nonnegative_integer_value_validator_hook('num_updates')],
                "required": True, "default": None
            },
            "dry_run": {
                "rules": [bool], "required": True, "default": False
            },
            "batch_maxnum": {
                "rules": [cls._nonnegative_integer_value_validator_hook('batch_maxnum')],
                "required": True, "default": None
            },
            "test_ratio": {
                "rules": [float, cls._test_ratio_hook], "required": False, "default": 0.0
            },
            "test_batch_size": {
                "rules": [cls.optional_type(typespec=int, argname='test_batch_size')],
                "required": False,
                "default": 0
            },
            "test_on_local_updates": {
                "rules": [bool], "required": False, "default": False
            },
            "test_on_global_updates": {
                "rules": [bool], "required": False, "default": False
            },
            "shuffle_testing_dataset": {
                "rules": [bool], "required": False, "default": False
            },
            "test_metric": {
                "rules": [cls._metric_validation_hook], "required": False, "default": None
            },
            "test_metric_args": {
                "rules": [dict], "required": False, "default": {}
            },
            "log_interval": {
                "rules": [int], "required": False, "default": 10
            },
            "fedprox_mu": {
                "rules": [cls._fedprox_mu_validator], 'required': False, "default": None
            },
            "use_gpu": {
                "rules": [bool], 'required': False, "default": False
            },
            "dp_args": {
                "rules": [cls._validate_dp_args], "required": True, "default": None
            },
            "share_persistent_buffers": {
                "rules": [bool], "required": False, "default": True
            },
            "random_seed": {
                "rules": [cls.optional_type(typespec=int, argname='random_seed')], "required": True, "default": None
            }
        }

    def __str__(self) -> str:
        """
        Display the Training_Args values as a string.

        Returns:
            printable version of TrainingArgs value
        """
        return str(self._ta)

    def __repr__(self) -> str:
        """
        Display the Training_Args full content for debugging purpose.

        Returns:
            printable version of TrainingArgs (scheme and value)
        """
        return f"scheme:\n{self._scheme}\nvalue:\n{self._ta}"

    def __setitem__(self, key: str, value: Any) -> Any:
        """
        Validate and then set a (key, value) pair to the current object.

        Args:
            key:   key
            value: value

        Returns:
            Full object updated with the validated (key, value) incorporated

        Raises:
            FedbiomedUserInputError: in case of problem (invalid key or value)
        """

        try:
            ta = deepcopy(self._ta)
            ta[key] = value
            self._sc.validate(ta)
            self._ta[key] = value  # only update it value is OK
        except (RuleError, ValidateError) as e:
            #
            # transform to FedbiomedError
            msg = ErrorNumbers.FB414.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)
        return deepcopy(self._ta[key])

    def __getitem__(self, key: str) -> Any:
        """
        Returns a copy of the value associated to a key.

        Args:
            key:   key

        Returns:
            value

        Raises:
            FedbiomedUserInputError: in case of bad key
        """
        try:
            ret = self._ta[key]
            return ret
        except KeyError:
            # transform to FedbiomedError
            msg = ErrorNumbers.FB414.value + f": The key `{key}` does not exist in training args"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

    def update(self, values: Dict) -> TypeVar("TrainingArgs"):
        """
        Update multiple keys of the training arguments.

        Args:
            values:  a dictionnary of (key, value) to validate/update

        Returns:
            the object itself after modification

        Raises:
            FedbiomedUserInputError: in case of bad key or value in values
        """
        for k in values:
            self.__setitem__(k, values[k])
        return self

    def __ixor__(self, other: Dict) -> TypeVar("TrainingArgs"):
        """
        Syntax sugar for update().

        **Usage:**
        ```python
        t = TrainingArgs()
        t ^= { 'epochs': 2 , 'lr': 0.01 }
        ```
        Args:
            other:  a dictionnary of keys to validate/update

        Returns:
            the object itself after modification

        Raises:
            FedbiomedUserInputError: in case of bad key or value in values
        """
        return self.update(other)

    def scheme(self) -> Dict:
        """
        Returns the scheme of a TrainingArgs instance.

        The scheme is not necessarily the default_scheme (returned by TrainingArgs.default_scheme().

        Returns:
            scheme:  the current scheme used for validation
        """
        return deepcopy(self._scheme)

    def default_value(self, key: str) -> Any:
        """
        Returns the default value for the key.

        Args:
            key:  key

        Returns:
            value: the default value associated to the key

        Raises:
            FedbiomedUserInputError: in case of problem (invalid key or value)
        """
        if key in self._sc.scheme():
            if "default" in self._sc.scheme()[key]:
                return deepcopy(self._sc.scheme()[key]["default"])
            else:
                msg = ErrorNumbers.FB410.value + \
                      f"no default value defined for key: {key}"
                logger.critical(msg)
                raise FedbiomedUserInputError(msg)
        else:
            msg = ErrorNumbers.FB410.value + \
                  f"no such key: {key}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

    def dict(self) -> dict:
        """Returns a copy of the training_args as a dictionary."""

        ta = deepcopy(self._ta)
        return ta

    def get_state_breakpoint(self):
        """Returns JSON serializable dict as state for breakpoints"""

        # TODO: This method is a temporary solution for JSON
        # serialize error during breakpoint save operation
        args = self.dict()
        test_metric = args.get('test_metric')

        if test_metric and isinstance(test_metric, MetricTypes):
            args['test_metric'] = test_metric.name

        return args

    @classmethod
    def load_state_breakpoint(cls, state: Dict) -> 'TrainingArgs':
        """Loads training arguments state"""
        if state.get('test_metric'):
            state.update(
                {'test_metric': MetricTypes.get_metric_type_by_name(
                                                state.get('test_metric'))})

        return cls(state)

    def get(self, key: str, default: Any = None) -> Any:
        """Mimics the get() method of dict, provided for backward compatibility.

        Args:
            key: a key for retrieving data fro the dictionary
            default: default value to return if key does not belong to dictionary
        """
        try:
            return deepcopy(self._ta[key])
        except KeyError:
            # TODO: test if provided default value is compliant with the scheme
            return default
