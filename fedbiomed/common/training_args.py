"""
Provide a way to easily to manage training arguments.
"""


from typing import Any, Dict, TypeVar

from fedbiomed.common.validator import SchemeValidator, ValidateError, RuleError


_E = TypeVar("Experiment")  # only for typing


class TrainingArgs():
    """
    Provide a container to deal with training arguments.

    More to come...
    """

    _grammar = {
        "lr": {
            "rules": [ float ],
            "required": True,
            "default": 0.01
        }
    }
    """
    Scheme which describes the expected keys and values for training arguments.
    """

    def __init__(self, ta: Dict):
        """
        Create a TrainingArgs from a Dict with input validation.

        Args:
            ta:  Dictionnary describing the TrainingArgs grammar.

        Raises:
            ValidateError: if ta is not valid
        """

        self._ta = ta
        try:
            self._sc = SchemeValidator(self._grammar)
        except RuleError:
            #
            # internal error (invalid grammar)
            raise

        # check user input
        try:
            self._sc.validate(ta)
        except (ValidateError):
            # transform to a Fed-BioMed error
            raise


    def __repr__(self):
        """
        Display the Training_Args content.
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
            self._ta[key] = value
            self._sc.validate(self._ta)
        except (RuleError, ValidateError):
            #
            # transform to FedbiomedError
            raise
        return self._ta[key]


    def modify(self, values: Dict) -> _E:
        """
        Modify multiple keys of the trainig arguments.
        """

        for k in values:
            self.__setitem__(k, values[k])

        return self
