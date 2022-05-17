""" All the fedbiomed errors/Exceptions """


# Do not import other fedbiomed package here to avoid dependency loop

class FedbiomedError(Exception):
    """
    Top class of all our exceptions.

    this allows to catch every Fedbiomed*Errors in a single except block
    """
    pass


# all inherited Errors

class FedbiomedAggregatorError(FedbiomedError):
    """
    Exception specific to the Aggregator classes/subclasses.
    """
    pass


class FedbiomedDataManagerError(FedbiomedError):
    """
    Exception for DataManager errors.
    """
    pass


class FedbiomedDatasetManagerError(FedbiomedError):
    """
    Exceptions specific for the class DatasetManager.
    """
    pass


class FedbiomedEnvironError(FedbiomedError):
    """
    Exception specific to the Environ class.
    """
    pass


class FedbiomedExperimentError(FedbiomedError):
    """
    Exception specific to the Experiment class.
    """
    pass


class FedbiomedLoggerError(FedbiomedError):
    """
    Exception specific to the Logger class.
    """
    pass


class FedbiomedMessageError(FedbiomedError):
    """
    Exception specific to the Message class, usually a badly formed message.
    """
    pass


class FedbiomedMessagingError(FedbiomedError):
    """
    Exception specific to the Messaging (communication) class.

    Usually a problem with the communication framework
    """
    pass


class FedbiomedMetricError(FedbiomedError):
    """
    Exception raised when evualution fails because of inconsistence in using the metric.
    """
    pass


class FedbiomedModelManagerError(FedbiomedError):
    """
    Exception specific to the ModelManager.

    (from fedbiomed.common.model_manager)
    """
    pass


class FedbiomedRepositoryError(FedbiomedError):
    """
    Exception of the `Repository` class.
    """
    pass


class FedbiomedResponsesError(FedbiomedError):
    """
    Exception specific to Responses class.
    """
    pass


class FedbiomedRoundError(FedbiomedError):
    """
    Exceptions specific for the class node round class.
    """
    pass


class FedbiomedSilentTerminationError(FedbiomedError):
    """
    Exception for silently terminating the researcher from a notebook.
    """
    def _render_traceback_(self):
        return []


class FedbiomedSkLearnDataManagerError(FedbiomedError):
    """
    Exceptions specific for the class SkLearnDataset.
    """
    pass


class FedbiomedStrategyError(FedbiomedError):
    """
    Exception specific to the Strategy class and subclasses.
    """
    pass


class FedbiomedTaskQueueError(FedbiomedError):
    """
    Exception specific to the internal queuing system.
    """
    pass


class FedbiomedTorchDataManagerError(FedbiomedError):
    """
    Exceptions specific for the class TorchDataset.
    """
    pass


class FedbiomedTorchTabularDatasetError(FedbiomedError):
    """
    Exceptions specific for the class fedbiomed.common.data.TorchTabularDataset.
    """
    pass


class FedbiomedTrainingError(FedbiomedError):
    """
    Exception raised then training fails.
    """
    pass


class FedbiomedTrainingPlanError(FedbiomedError):
    """
    Exception specific to errors while getting source of the model class.
    """
    pass


class FedbiomedUserInputError(FedbiomedError):
    """
    Exception raised then user input is invalid.
    """
    pass
