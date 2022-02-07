'''
all the fedbiomed errors

do not import other fedbiomed package here to avoid dependancy loop
'''


class FedbiomedError(Exception):
    """
    top class of all our exceptions

    this allows to catch every Fedbiomed*Errors in a single except block
    """
    pass

class FedbiomedEnvironError(FedbiomedError):
    """
    Exception specific to the Environ class
    """
    pass

class FedbiomedLoggerError(FedbiomedError):
    """
    Exception specific to the Logger class
    """
    pass

class FedbiomedMessageError(FedbiomedError):
    """
    Exception specific to the Message class
    usually a badly formed message
    """
    pass

class FedbiomedStrategyError(FedbiomedError):
    """
    Exception specific to the Strategy class and subclasses
    """
    pass


class FedbiomedTrainingError(FedbiomedError):
    """
    Exception raises then training (researcher/node) class
    """
    pass
