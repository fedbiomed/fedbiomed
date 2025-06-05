# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

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


class FedbiomedCertificateError(FedbiomedError):
    """
    Certificate error
    """
    pass


class FedbiomedCommunicationError(FedbiomedError):
    """
    Fedbiomed errors related to gRPC communication
    """
    pass


class FedbiomedConfigurationError(FedbiomedError):
    """
    Exception specific to the Config classes.
    """
    pass


class FedbiomedDataLoadingPlanError(FedbiomedError):
    """
    Exceptions specific for the class fedbiomed.common.data.DataLoadingPlan.
    """
    pass


class FedbiomedDataLoadingPlanValueError(FedbiomedError):
    """
    Exceptions similar to Value Error for a DataLoadingPlan.
    """
    pass


class FedbiomedDatasetError(FedbiomedError):
    """
    Generic exception for a Dataset class.
    """
    pass


class FedbiomedDatasetValueError(FedbiomedError):
    """
    ValueErrors raised by any Dataset class.
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


class FedbiomedDPControllerError(FedbiomedError):
    """
    Exceptions specific for the class DPController
    """
    pass


class FedbiomedExperimentError(FedbiomedError):
    """
    Exception specific to the Experiment class.
    """
    pass


class FedbiomedFederatedDataSetError(FedbiomedError):
    """
    Exception specific to the FederatedDataSetError class.
    """
    pass


class FedbiomedLoadingBlockError(FedbiomedError):
    """
    Exception specific to the DataLoadingBlock classes/subclasses.
    """
    pass


class FedbiomedLoadingBlockValueError(FedbiomedError):
    """
    Exception similar to ValueError for a DataLoadingBlock.
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


class FedbiomedNodeStateAgentError(FedbiomedError):
    """
    Error in Node State Agent
    """
    pass


class FedbiomedNodeStateManagerError(FedbiomedError):
    """
    Error in Node State Manager
    """
    pass


class FedbiomedNodeToNodeError(FedbiomedError):
    """
    Error in Node to Node communications
    """
    pass


class FedbiomedOptimizerError(FedbiomedError):
    """
    Exception raised when an error is encountered within `Optimizer` code.
    """
    pass


class FedbiomedRoundError(FedbiomedError):
    """
    Exceptions specific for the node round class.
    """
    pass


class FedbiomedModelError(FedbiomedError):
    """
    Exceptions triggered from Model class
    """
    pass


class FedbiomedSecaggError(FedbiomedError):
    """
    Exceptions specific for the researcher secure aggregation class.
    """
    pass


class FedbiomedSecaggCrypterError(FedbiomedError):
    """
    Secure aggregation encryption error
    """


class FedbiomedSecureAggregationError(FedbiomedError):
    """
    Secure aggregation error
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


class FedbiomedSynchroError(FedbiomedError):
    """
    Error in synchro objects
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


class FedbiomedTrainingPlanSecurityManagerError(FedbiomedError):
    """
    Exception specific to the TrainingPlanSecurityManager.

    (from fedbiomed.common.model_manager)
    """
    pass


class FedbiomedTypeError(FedbiomedError, TypeError):
    """
    TypeError for Fed-BioMed
    """
    pass


class FedbiomedUserInputError(FedbiomedError):
    """
    Exception raised then user input is invalid.
    """
    pass


class FedbiomedValueError(FedbiomedError, ValueError):
    """
    ValueError for Fed-BioMed
    """
    pass


class FedbiomedVersionError(FedbiomedError):
    """
    Error in the versions of one of Fed-BioMed's components
    """
