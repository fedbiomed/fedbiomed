"""
Provide a way to easily to manage training arguments.
"""


from copy import deepcopy
from typing import Any, Dict, TypeVar, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedUserInputError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.validator import SchemeValidator, ValidateError, \
    RuleError, validator_decorator


_MyOwnType = TypeVar("TrainingArgs")
"""
Used for type checking
"""

class TrainingArgs():
    """
    Provide a container to deal with training arguments.

    More to come...
    """
    def __init__(self, ta: Dict = None, extra_scheme: Dict = None, only_required: bool = True):
        """
        Create a TrainingArgs from a Dict with input validation.

        Args:
            ta:     dictionnary describing the TrainingArgs scheme.
                    if empty dict or None, a minimal instance of TrainingArgs
                    will be initialized with default values for required keys
            scheme: user provided scheme extension, which add new rules or
                    update the scheme of the default training args.
                    Warning: this is a dangerous featuret, provided to
                    developpers, to ease the test of future Fed-Biomed features

        Raises:
            ValidateError: if ta is not valid
            RuleError: if extra_scheme is not valid
        """
        self._scheme = TrainingArgs.default_scheme()

        if extra_scheme is None:
            extra_scheme = {}

        for k in extra_scheme:
            self._scheme[k] = extra_scheme[k]

        try:
            self._sc = SchemeValidator(self._scheme)
        except RuleError:
            #
            # internal error (invalid scheme)
            raise

        # scheme is validated from here
        if ta is None:
            ta = {}

        try:
            self._ta = self._sc.populate_with_defaults( ta,
                                                        only_required = only_required)
        except RuleError:
            # scheme has required keys without defined default value
            raise

        # finaly check user input
        try:
            self._sc.validate(self._ta)
        except (ValidateError):
            # transform to a Fed-BioMed error
            raise


    # validators
    @staticmethod
    @validator_decorator
    def _metric_validation_hook( metric: Union[MetricTypes, str, None] ):
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
    def _test_ratio_hook( v: Any):
        """
        Test if in [ 0.0 , 1.0]  interval.
        """
        if v < 0 or v > 1:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _loss_rate_hook( v: Any):
        """
        Test if in [ 0.0 , 1.0]  interval.
        """
        if v < 0:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _always_true_hook( v: Any):
        """
        As is says
        """
        return True

    @classmethod
    def default_scheme(cls) -> Dict:
        """
        Returns the default (base) scheme for TrainingArgs.
        """
        return  {
            # loss rate
            "lr": {
                "rules": [ float, cls._loss_rate_hook ],
                "required": False,
#                "default": 0.01
            },

            # batch_size
            "batch_size": {
                "rules": [ int ],
                "required": False,
#                "default": 48
            },

            # epochs
            "epochs": {
                "rules": [ int ],
                "required": False,
#                "default": 1
            },

            # dry_run
            "dry_run": {
                "rules": [ bool ],
                "required": False,
#                "default": False
            },

            # batch_maxnum
            "batch_maxnum": {
                "rules": [ int ],
                "required": False,
#                "default": 100
            },

            # test_ratio
            "test_ratio": {
                "rules": [ float, cls._test_ratio_hook ],
                "required": False,
                "default": 0.0
            },

            # test_on_local_updates
            "test_on_local_updates": {
                "rules": [ bool ],
                "required": False,
                "default": False
            },

            # tests_on_globals_updates
            "test_on_global_updates": {
                "rules": [ bool ],
                "required": False,
                "default": False
            },

            # test_metric
            "test_metric": {
                "rules": [ cls._metric_validation_hook ],
                "required": False,
                "default": None
            },

            # test_metric_args (no test)
            "test_metric_args": {
                "rules": [ dict ],
                "required": False,
                "default": {}
            }

        }



    def __repr__(self) -> str:
        """
        Display the Training_Args content.

        Returns:
            printable version of TrainingArgs
        """
        return str(self._ta)


    def __setitem__(self, key: str, value: Any) -> Any:
        """
        Validate and then set a value for a given key.

        Args:
            key:   key
            value: values

        Returns:
            validated keys

        Raises:
            ValidateError: in case of problem (invalid key or value)
        """

        try:
            ta = deepcopy(self._ta)
            ta[key] = value
            self._sc.validate(ta)
            self._ta[key] = value  # only update it value is OK
        except (RuleError, ValidateError) as e:
            #
            # transform to FedbiomedError
            msg = ErrorNumbers.FB410.value + \
                f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)
        return self._ta[key]


    def __getitem__(self, key: str) -> Any:
        """
        Returns the value associated to a key.

        Args:
            key:   key

        Returns:
            value

        Raises:
            KeyError: if key does not exist
        """
        return self._ta[key]


    def update(self, values: Dict) -> _MyOwnType:
        """
        Update multiple keys of the trainig arguments.

        Args:
            values:  a dictionnary of keys to validate/update

        Return:
            self: the object itself after modification

        Raises:
            RuleError: if a key in invalid
            ValidateError: if a value is invalid
        """
        for k in values:
            self.__setitem__(k, values[k])
        return self


    def __ixor__(self, other: Dict) -> _MyOwnType:
        """
        Syntax sugar for update().

        Args:
            other:  a dictionnary of keys to validate/update

        Return:
            self: the object itself after modification

        Raises:
            RuleError: if a key in invalid
            ValidateError: if a value is invalid
        """
        return self.update(other)


    def scheme(self) -> Dict:
        """
        Returns the scheme of a TrainingArgs instance.

        The scheme is not necessarly the [`default_scheme`][default_scheme]

        Returns:
            scheme:  the current scheme used for validation
        """
        return self._scheme


    def default_value(self, key: str) -> Any:
        """
        Returns the default value for the key.

        Args:
            key:  key

        Returns:
            value: the default value associated to the key

        Raises:
            KeyError:  if key is not part of the sheme or
            ValueError: if no default value is defined for the key
        """
        if key in self._sc.scheme():
            if "default" in self._sc.scheme()[key]:
                return self._sc.scheme()[key]["default"]
            else:
                raise ValueError(f"no default value defined for key: {key}")
        else:
            raise KeyError(f"no such key: {key}")


    def dict(self):
        """Returns the training_args as a dictionnary."""
        return self._ta
