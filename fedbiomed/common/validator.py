'''
Provide a Validator class for validating
parameters against a rule.

Rule can be simple (type to check againt, function)
or can be declared as a json grammar to validate json data.
'''


import functools
import inspect

from enum import Enum

from fedbiomed.common.logger import logger


class _ValidatorHookType(Enum):
    """
    List of all method available to execute a validation hook
    """
    INVALID = 1
    TYPECHECK = 2
    FUNCTION = 3
    SCHEME_VALIDATOR = 4
    SCHEME_AS_A_DICT = 5


def validator_decorator(func):
    '''
    function decorator for simplifying the writing of validator hook
    (aka a fvalidation function)

    The decorator catches the output of the validator hook and build
    a tuple (boolean, string) as expected by the Validator class:

    It creates an error message if not provided by the decorated function
    The error message is forced to if the decorated function returns True

    Args:
       function to decorate

    Returns:
       decorated function
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # execute the wrapped function
        status = func(*args, **kwargs)

        # we expect a tuple [ bolean, str] as output of func()
        # but we try to be resilient to function that simply return boolean
        error = "validation error then calling: " + func.__name__
        if isinstance(status, tuple):
            status, *error = status

        if status:
            return status, None
        else:
            return status, str(error)
    return wrapper


class SchemeValidator(object):
    """
    validation class for scheme (grammar) which describes json data

    this class uses Validator base functions
    """

    # necessary keys and key types
    _necessary = { 'rules': list }  # list of callable, str, class

    # optionnal keys (no type associated to default)
    # even if the default may be checked againt the list of the rules
    _optionnal = { 'default': None, 'required': bool }


    def __init__(self, scheme):
        """
        validate a scheme describing a json data
        against SchemeValidator rules

        Args:
            scheme     scheme to validate
        """

        status = self.__validate_scheme(scheme)

        if isinstance(status, bool) and status:
            self._scheme = scheme
            self._is_valid = True

        else:
            self._scheme = None
            self._is_valid = False
            logger.error("scheme is not valid: " + status)


    def validate(self, value):
        """
        validate a value against the scheme passed at creation time
        """
        if not self.is_valid():
            return "scheme is not valid"

        # check the value against the scheme
        for k, v in self._scheme.items():
            if 'required' in v and v['required'] is True and k not in value:
                return False, str(k) + " value is required"

        for k in value:
            if k not in self._scheme:
                return False, "undefined key (" + str(k) + ") in scheme"

            for hook in self._scheme[k]['rules']:
                if not Validator().validate(value[k], hook):
                    return False, "invalid value (" + str(value[k]) + ") for key: " + str(k)

        return True



    def __validate_scheme(self, scheme):
        """
        scheme validator function (internal)

        Args:
            scheme    JSON describing a scheme

        Returns:
            True      (bool) if everything is OK
            error_msg (str)  in case of error
        """

        if not isinstance(scheme, dict) or len(scheme) == 0:
            return("validator scheme must be a non empty dict")

        for n in self._necessary:
            for key in scheme:

                if not isinstance( scheme[key], dict) or len(scheme[key]) == 0 :
                    return("validator rule of (" + \
                           str(key) + \
                           ") scheme must be a non empty dict")

                if n not in scheme[key]:
                    return("required subkey (" + \
                           str(n) + \
                           ") is missing for key: " + \
                           str(key)
                        )

                value_in_scheme = scheme[key][n]
                requested_type  = self._necessary[n]
                if requested_type is not None and \
                   not isinstance(value_in_scheme, requested_type):

                    return("bad type for subkey (" + \
                           str(n) + \
                           ") for key: " + \
                           str(key)
                           )

                # special case for 'rules'
                if not n == 'rules':
                    continue

                # check that rules contains valid keys for Validator

                # TODO: should use Validator._is_hook_type_valid
                for element in scheme[key][n]:

                    if isinstance(element, str) and not self.is_known_rule(element):
                        return("rule (" + \
                               str(element) + \
                               ") for key: " + \
                               str(key) + \
                               " is not defined"
                               )

                    if not inspect.isclass(element) and \
                       not inspect.isfunction(element) and \
                       not isinstance(element, str):
                        return("bad content for subkey (" + \
                               str(n) + \
                               ") for key: " + \
                               str(key)
                               )

        # check that all provided keys of scheme are accepted
        for key in scheme:
            for subkey in scheme[key]:
                if subkey not in self._necessary and subkey not in self._optionnal:
                    return ("unknown subkey (" + \
                            str(subkey) + \
                            ") provided for key: " + \
                            str(key)
                            )

        # scheme is validated
        return True


    def is_valid(self):
        """
        status of the scheme passed at creation time
        """
        return ( self._scheme is not None ) or self._is_valid

    def scheme(self):
        """
        return the (current) scheme
        """
        return self._scheme or None


class Validator(object):
    """
    container class for validation functions

    Validator provides several ways to trigger validations:
    - registering/using rules (accessible via thier name)
    - providing a simple function to execute
    - checking the value type
    """

    _validation_rulebook = {}

    def __init__(self):
        pass


    def validate(self, value, rule, strict = True):
        '''
        check value against a:
        - (registered) rule
        - a provided function,
        - a simple type checking
        - a SchemeValidator
        '''

        # rule is in the rulebook -> execute the rule associated function
        if isinstance(rule, str) and rule in self._validation_rulebook:

            status, error = Validator._hook_execute(value,
                                                    self._validation_rulebook[rule])
            if not status:
                logger.error(error)
            return status

        # rule is an unknown string
        if isinstance(rule, str):
            if strict:
                logger.error("unknown rule: " + str(rule))
                return False
            else:
                logger.warning("unknown rule: " + str(rule))
                return True

        # consider the rule as a direct rule definition
        status, error = Validator._hook_execute(value, rule)

        if not status:
            logger.error(error)
        return status



    @staticmethod
    def _hook_type(hook):
        """
        Return the hook type

        Args:
            hook   a hook to validate

        Returns:
            enum   return the method associated with this hook
        """

        if isinstance(hook, SchemeValidator):
            return _ValidatorHookType.SCHEME_VALIDATOR

        if isinstance(hook, dict):
            return _ValidatorHookType.SCHEME_AS_A_DICT

        if inspect.isclass(hook):
            return _ValidatorHookType.TYPECHECK

        if callable(hook):
            return _ValidatorHookType.FUNCTION

        # not valid
        return _ValidatorHookType.INVALID

    @staticmethod
    def _is_hook_type_valid(hook):
        """
        verify that the hook type associated to a rule is valid

        it does not validate the hook for function and SchemeValidator,
        it only verifies that the hook can be registered for later use

        Args:
            hook   a hook to validate

        Returns:
            enum   return the method associated with this hook
        """

        hook_type = Validator._hook_type(hook)

        if hook_type == _ValidatorHookType.INVALID:
            return False
        else:
            return True

    @staticmethod
    @validator_decorator
    def _hook_execute(value, hook):
        """
        execute the test associated with the hook on the value

        Args:
            value   to test
            hook    to tests against
            strict  boolen to decide is the test is strcit or not

        Returns:
            (boolean, string)  result of the test and optionnal error message

        """
        hook_type = Validator._hook_type(hook)

        if hook_type is _ValidatorHookType.INVALID:
            return False, "hook is not authorized"

        if hook_type is _ValidatorHookType.TYPECHECK:
            status = isinstance(value, hook)
            return status, "wrong input: " + str(value) + " should be a " + str(hook)

        if hook_type is _ValidatorHookType.FUNCTION:
            return hook(value)

        if hook_type is _ValidatorHookType.SCHEME_VALIDATOR:
            return hook.validate(value)

        if hook_type is _ValidatorHookType.SCHEME_AS_A_DICT:
            sc = SchemeValidator( hook )
            if not sc.is_valid():
                return False, "scheme is not valid"

            return sc.validate(value)


    def rule(self, rule):
        '''
        return validator for the rule (if registered)
        '''
        if rule in self._validation_rulebook:
            return self._validation_rulebook[rule]
        else:
            return None

    def is_known_rule(self, rule):
        '''
        return True if the rule is registered
        '''
        return (rule in self._validation_rulebook)

    def register_rule(self, rule, hook, override = False):
        '''
        add a rule/validation_function to the rulebook
        '''
        if not isinstance(rule, str):
            logger.error("rule name must be a string")
            return False

        if not override and rule in self._validation_rulebook:
            logger.warning("validator already register for rule: " + rule)
            return False

        if not Validator._is_hook_type_valid(hook):
            logger.error("action associated to the rule is unallowed")
            return False

        # hook is a dict, we transform it to a SchemeValidator
        if isinstance(hook, dict):
            sv = SchemeValidator( hook )
            if not sv.is_valid():
                return False
            else:
                hook = sv

        self._validation_rulebook[rule] = hook
        return True

    def delete_rule(self, rule):
        '''
        delete a rule/validation_function from the rulebook
        '''
        if rule in self._validation_rulebook:
            del self._validation_rulebook[rule]
