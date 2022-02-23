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


class FedbiomedMessagingError(FedbiomedError):
    """
    Exception specific to the Messaging (communication) class
    usually a problem with the communication framework
    """
    pass


class FedbiomedTaskQueueError(FedbiomedError):
    """
    Exception specific to the internal queuing system
    """
    pass


class FedbiomedStrategyError(FedbiomedError):
    """
    Exception specific to the Strategy class and subclasses
    """
    pass


class FedbiomedTrainingError(FedbiomedError):
    """
    Exception raised then training fails
    """
    pass


class FedbiomedExperimentError(FedbiomedError):
    """
    Exception specific to the Experiment class
    """
    pass


# specific exception
class FedbiomedSilentTerminationError(FedbiomedError):
    """
    Exception for silently terminating the researcher from a notebook
    """

    def _render_traceback_(self):
        pass


class FedbiomedResponsesError(FedbiomedError):
    """
    Exception specific to Responses class
    """
    pass


class FedbiomedRepositoryError(FedbiomedError):
    """
    Exception of the `Repository` class
    """
    pass


class FedbiomedTrainingPlanError(FedbiomedError):
    """
    Exception specific to errors while getting source of the model class
    """
    pass
