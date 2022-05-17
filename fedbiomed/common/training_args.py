"""
Provide a way to easily to manage training arguments.
"""


from copy import deepcopy
from typing import Any, Dict, TypeVar, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedUserInputError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.validator import SchemeValidator, ValidatorError, \
    ValidateError, RuleError, validator_decorator


class TrainingArgs():
    """
    Provide a container to manage training arguments.

    This class uses the Validator and SchemeValidator classes
    and provides a default scheme, which describes the arguments
    necessary to train/test a TrainingPlan.

    It also permits to extend the TrainingArgs then testing new features
    by supplying an extra_scheme at TraininfArgs instanciation.
    """
    def __init__(self, ta: Dict = None, extra_scheme: Dict = None, only_required: bool = True):
        """
        Create a TrainingArgs from a Dict with input validation.

        Args:
            ta:     dictionnary describing the TrainingArgs scheme.
                    if empty dict or None, a minimal instance of TrainingArgs
                    will be initialized with default values for required keys
            extra_scheme: user provided scheme extension, which add new rules or
                    update the scheme of the default training args.
                    Warning: this is a dangerous feature, provided to
                    developpers, to ease the test of future Fed-Biomed features
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
            msg = ErrorNumbers.FB410.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

        # scheme is validated from here
        if ta is None:
            ta = {}

        try:
            self._ta = self._sc.populate_with_defaults( ta,
                                                        only_required = only_required)
        except ValidatorError as e:
            # scheme has required keys without defined default value
            msg = ErrorNumbers.FB410.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)

        # finally check user input
        try:
            self._sc.validate(self._ta)
        except (ValidateError) as e:
            # transform to a Fed-BioMed error
            msg = ErrorNumbers.FB410.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)


    # validators
    @staticmethod
    @validator_decorator
    def _metric_validation_hook( metric: Union[MetricTypes, str, None] ) -> Union[bool, str]:
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
    def _test_ratio_hook( v: Any) -> bool:
        """
        Test if in [ 0.0 , 1.0]  interval.
        """
        if v < 0 or v > 1:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _lr_hook( v: Any):
        """
        Test if lr is greater than 0.
        """
        if v < 0:
            return False
        else:
            return True

    @classmethod
    def default_scheme(cls) -> Dict:
        """
        Returns the default (base) scheme for TrainingArgs.
        """
        return {
            # lr
            "lr": {
                "rules": [ float, cls._lr_hook ],
                "required": False,
#   "default": 0.01
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
        Validate and then set a value for a given key.

        Args:
            key:   key
            value: values

        Returns:
            validated keys

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
            msg = ErrorNumbers.FB410.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)
        return deepcopy(self._ta[key])


    def __getitem__(self, key: str) -> Any:
        """
        Returns the value associated to a key.

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
        except (KeyError) as e:
            #
            # transform to FedbiomedError
            msg = ErrorNumbers.FB410.value + f": {e}"
            logger.critical(msg)
            raise FedbiomedUserInputError(msg)


    def update(self, values: Dict) -> TypeVar("TrainingArgs"):
        """
        Update multiple keys of the training arguments.

        Args:
            values:  a dictionnary of keys to validate/update

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
            other (Dict):  a dictionnary of keys to validate/update

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
            key (str):  key

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


    def dict(self):
        """Returns a copy of the training_args as a dictionary."""

        ta = deepcopy(self._ta)
        if 'test_metric' in ta and \
           isinstance(ta['test_metric'], MetricTypes):
            # replace MetricType value by a string
            ta['test_metric'] = ta['test_metric'].name
        return ta


    def get(self, key: str, default: Any = None) -> Any:
        """Mimics the get() method of dict, provided for backward compatibility.

        Args:
            key (str): a key for retrieving data fro the dictionary
            default (Any): default value to return if key does not belong to dictionary
        """
        try:
            return deepcopy(self._ta[key])
        except KeyError:
            return default
