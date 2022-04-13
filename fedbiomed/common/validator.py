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
        check value against a (registered) rule, or a provided function,
        or a simple type checking
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


    def rule(self, rule):
        '''
        return validator for the rule (if registered)
        '''
        if rule in self._validation_rulebook:
            return self._validation_rulebook[rule]
        else:
            return None

    def knows_rule(self, rule):
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
            logger.warning("validator already register for " + rule)
            return False
        else:
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

                    if isinstance(element, str) and not self.knows_rule(element):
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
