'''
Provide a Validator class for validating
parameters against a rule.

Rule can be simple (type to check againt, function)
or can be declared as a json grammar to validate json data.
'''


import functools
import inspect

from fedbiomed.common.logger import logger


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
            return "scheme is not conform"

        # check the value against the scheme
        for k, v in self._scheme.items():
            if 'required' in v and v['required'] is True and k not in value:
                return False, str(k) + " value is required"

        for k in value:
            if k not in self._scheme and strict:
                return False, "undefined key (" + str(k) + ") in scheme"

            for hook in self._scheme[k]['rules']:
                if not self.validate(value[k], hook, strict):
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

        # rule is a dict -> scheme validation
        if isinstance(rule, dict):

            status, error = self._scheme_validator_hook(value, rule, strict)
            if not status:
                logger.error(str(error))
                return False
            return True

        # rule is in the rulebook -> execute the rule associated function
        if rule in self._validation_rulebook:
            status, error = self._validation_rulebook[rule](value)
            if not status:
                logger.error("wrong input for rule: " + rule + " - " + error + " ( " + str(value) + " )" )
            return status

        # rule is a python Class -> type checking
        if inspect.isclass(rule):
            status = isinstance(value, rule)
            if not status:
                logger.error("wrong input: " + str(value) + " should be a " + str(rule))
            return status


        # rule has been passed as a function -> execute the function
        if inspect.isfunction(rule):
            status = rule(value)
            error = ""

            if isinstance(status, tuple):
                status, *error = status

            if not status:
                logger.error("wrong input: " + str(value) + " " + str(error))
            return status

        # undefined rule
        if strict:
            logger.error("unknown rule: " + str(rule))
            return False
        else:
            logger.warning("unknown rule: " + str(rule))

        # but we are gentle
        return True


    def _execute_rule(self, value, rule):
        """
        execute the rule registered in the rulebook.
        The way is is executed depends on what is registered

        Args:
            value  value to test
            rule   name of the rule (str)
        """

        if rule not in self._validation_rulebook:
            return False

        action = self._validation_rulebook[rule]


    @staticmethod
    def _is_hook_type_valid(hook):
        """
        verify that the hook type associated to a rule is valid

        it does not validate the hook for function and SchemeValidator,
        it only verifies that the hook can be registered for later use
        """

        if isinstance(hook, SchemeValidator):
            # SchemeValidator
            return True

        if isinstance(hook, dict):
            # SchemeValidator
            return True

        if isinstance(hook, type):
            # TypeChecking
            return True

        if callable(hook):
            # function"
            return True

        # not valid
        return False

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


    @validator_decorator
    def _scheme_validator_hook(self, json, scheme, strict):
        """
        provide validator for Json scheme description
        """

        if not isinstance(scheme, dict) or len(scheme) == 0:
            return False, "validator scheme must be a non empty dict"

        # necessary keys and key types
        necessary = { 'rules': list }  # list of callable, str, class

        # optionnal keys (no type associated to default)
        # even if the default may be checked againt the list of the rules
        optionnal = { 'default': None, 'required': bool }

        for n in necessary:
            for key in scheme:

                if not isinstance( scheme[key], dict) or len(scheme[key]) == 0 :
                    return False, "validator rule of (" + str(key) + ") scheme must be a non empty dict"

                if n not in scheme[key]:
                    return False, "required subkey (" + \
                        str(n) + \
                        ") is missing for key: " + \
                        str(key)

                value_in_scheme = scheme[key][n]
                requested_type  = necessary[n]
                if requested_type is not None and \
                   not isinstance(value_in_scheme, requested_type):
                    return False, "bad type for subkey (" + \
                        str(n) + \
                        ") for key: " + \
                        str(key)

                # special case for 'rules'
                if not n == 'rules':
                    continue

                # check that rules contains valid keys for Validator
                for element in scheme[key][n]:

                    if isinstance(element, str) and not self.is_known_rule(element):
                        return False, "rule (" + \
                            str(element) + \
                            ") for key: " + \
                            str(key) + \
                            " is not defined"

                    if not inspect.isclass(element) and \
                       not inspect.isfunction(element) and \
                       not isinstance(element, str):
                        return False, "bad content for subkey (" + \
                            str(n) + \
                            ") for key: " + \
                            str(key)

        # check that all provided keys of scheme are accepted
        for key in scheme:
            for subkey in scheme[key]:
                if subkey not in necessary and subkey not in optionnal:
                    return False, "unknown subkey (" + \
                        str(subkey) + \
                        ") provided for key: " + \
                        str(key)

        # scheme is validated
        # check the json against the scheme
        for k, v in scheme.items():
            if 'required' in v and v['required'] is True and k not in json:
                return False, str(k) + " value is required"

        for k in json:
            if k not in scheme and strict:
                return False, "undefined key (" + str(k) + ") in scheme"

            for hook in scheme[k]['rules']:
                if not self.validate(json[k], hook, strict):
                    return False, "invalid value (" + str(json[k]) + ") for key: " + str(k)

        return True
