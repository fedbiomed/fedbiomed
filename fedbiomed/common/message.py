# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
Definition of messages exchanged by the researcher and the nodes
'''

import functools

from dataclasses import dataclass
from typing import Any, Callable, Dict

from fedbiomed.common.constants import ErrorNumbers, __messaging_protocol_version__
from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.common.logger import logger


def catch_dataclass_exception(cls: Callable):
    """Encapsulates the __init__() method of dataclass in order to transform the exceptions sent
    by the dataclass (TypeError) into our own exception (FedbiomedMessageError)

    Args:
        cls: Dataclass to validate
    """

    def __cde_init__(self: Any, *args: list, **kwargs: dict):
        """This is the __init__() replacement.

        Its purpose is to catch the TypeError created by the __init__
        method of the @dataclass decorator and replace this exception by  FedbiomedMessageError

        Raises:
          FedbiomedMessageError if number/type of arguments is wrong
        """

        try:
            self.__class__.__dict__['__initial_init__'](self, *args, **kwargs)

        except TypeError as e:
            # this is the error raised by dataclass if number of parameter is wrong
            _msg = ErrorNumbers.FB601.value + ": bad number of parameters: " + str(e)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

    @functools.wraps(cls)
    def wrap(cls: Callable):
        """ Wrapper to the class given as parameter

        Class wrapping should keep some attributes (__doc__, etc) of the initial class or the API documentation tools
        will be mistaken

        """
        cls.__initial_init__ = cls.__init__
        setattr(cls, "__init__", __cde_init__)

        return cls

    return wrap(cls)


class Message(object):
    """Base class for all fedbiomed messages providing all methods
    to access the messages

    The subclasses of this class will be pure data containers (no provided functions)
    """

    def __post_init__(self):
        """ Post init of dataclass

        - remark: this is not check by @dataclass

        Raises:
            FedbiomedMessageError: (FB601 error) if parameters of bad type

        """

        if not self.__validate(self.__dataclass_fields__.items()):
            _msg = ErrorNumbers.FB601.value + ": bad input value for message: " + self.__str__()
            logger.critical(_msg)
            raise FedbiomedMessageError(_msg)

    def get_param(self, param: str):
        """Get the value of a given param

        Args:
            param: name of the param
        """
        return getattr(self, param)

    def get_dict(self) -> Dict[str, Any]:
        """Returns pairs (Message class attributes name, attributes values) into a dictionary

        Returns:
            Message as dictionary
        """
        return self.__dict__

    def __validate(self, fields: Dict[str, Any]) -> bool:
        """Checks whether incoming field types match with attributes class type.

        Args:
            fields: incoming fields

        Returns:
            If validated, ie everything matches, returns True, else returns False.
        """
        ret = True
        for field_name, field_def in fields:
            value = getattr(self, field_name)
            if not isinstance(value, field_def.type):
                logger.critical(f"{field_name}: '{value}' instead of '{field_def.type}'")
                ret = False
        return ret


#
# messages definition, sorted by
# - alphabetic order
# - Request/Reply regroupemnt
#

# AddScalar message
@dataclass
class RequiresProtocolVersion:
    """Mixin class for messages that must be endowed with a version field.

    Attributes:
        protocol_version: version of the messaging protocol used
    """
    protocol_version: str

@catch_dataclass_exception
@dataclass
class AddScalarReply(Message, RequiresProtocolVersion):
    """Describes a add_scalar message sent by the node.

    Attributes:
        researcher_id: ID of the researcher that receives the reply
        job_id: ID of the Job that is sent by researcher
        train: Declares whether scalar value is for training
        test: Declares whether scalar value is for validation
        test_on_local_updates: Declares whether validation is performed over locally updated parameters
        test_on_global_updates: Declares whether validation is performed over aggregated parameters
        metric: Evaluation metroc
        epoch: Scalar is received at
        total_samples: Number of all samples in dataset
        batch_samples: Number of samples in batch
        num_batches: Number of batches in single epoch
        iteration: Scalar is received at
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed

    """
    researcher_id: str
    node_id: str
    job_id: str
    train: bool
    test: bool
    test_on_local_updates: bool
    test_on_global_updates: bool
    metric: dict
    epoch: (int, type(None))
    total_samples: int
    batch_samples: int
    num_batches: int
    num_samples_trained: (int, type(None))
    iteration: int
    command: str


# Approval messages


@catch_dataclass_exception
@dataclass
class ApprovalRequest(Message, RequiresProtocolVersion):
    """Describes the TrainingPlan approval request from researcher to node.

    Attributes:
        researcher_id: id of the researcher that sends the request
        description: description of the training plan
        sequence: (unique) sequence number which identifies the message
        training_plan_url: URL where TrainingPlan is available
        command: request command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    description: str
    sequence: int
    training_plan_url: str
    command: str


@catch_dataclass_exception
@dataclass
class ApprovalReply(Message, RequiresProtocolVersion):
    """Describes the TrainingPlan approval reply (acknoledge) from node to researcher.

    Attributes:
        researcher_id: Id of the researcher that will receive the reply
        node_id: Node id that replys the request
        sequence: sequence number of the corresponding request
        status: status code received after uploading the training plan (usually HTTP status)
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    sequence: int
    status: int
    command: str
    success: bool


# Error message

@catch_dataclass_exception
@dataclass
class ErrorMessage(Message, RequiresProtocolVersion):
    """Describes an error message sent by the node.

    Attributes:
        researcher_id: ID of the researcher that receives the error message
        node_id: ID of the node that sends error message
        errnum: Error ID/Number
        extra_msg: Additional message regarding the error
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    errnum: ErrorNumbers
    extra_msg: str
    command: str


# List messages

@catch_dataclass_exception
@dataclass
class ListRequest(Message, RequiresProtocolVersion):
    """Describes a list request message sent by the researcher to nodes in order to list datasets belonging to
    each node.

    Attributes:
        researcher_id: Id of the researcher that sends the request
        command: Request command string

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    command: str


@catch_dataclass_exception
@dataclass
class ListReply(Message, RequiresProtocolVersion):
    """This class describes a list reply message sent by the node that includes list of datasets. It is a
    reply for ListRequest message from the researcher.

    Attributes:
        researcher_id: Id of the researcher that sends the request
        succes: True if the node process the request as expected, false if any exception occurs
        databases: List of datasets
        node_id: Node id that replys the request
        count: Number of datasets
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    success: bool
    databases: list
    node_id: str
    count: int
    command: str


# Log message

@catch_dataclass_exception
@dataclass
class LogMessage(Message, RequiresProtocolVersion):
    """Describes a log message sent by the node.

    Attributes:
        researcher_id: ID of the researcher that will receive the log message
        node_id: ID of the node that sends log message
        level: Log level
        msg: Log message
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    level: str
    msg: str
    command: str


# TrainingPlanStatus messages

@catch_dataclass_exception
@dataclass
class TrainingPlanStatusRequest(Message, RequiresProtocolVersion):
    """Describes a training plan approve status check message sent by the researcher.

    Attributes:
        researcher_id: Id of the researcher that sends the request
        job_id: Job id related to the experiment.
        training_plan_url: The training plan that is going to be checked for approval
        command: Request command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
   """

    researcher_id: str
    job_id: str
    training_plan_url: str
    command: str


@catch_dataclass_exception
@dataclass
class TrainingPlanStatusReply(Message, RequiresProtocolVersion):
    """Describes a training plan approve status check message sent by the node

    Attributes:
        researcher_id: Id of the researcher that sends the request
        node_id: Node id that replys the request
        job_id: job id related to the experiment
        succes: True if the node process the request as expected, false
            if any execption occurs
        approval_obligation : Approval mode for node. True, if training plan approval is enabled/required
            in the node for training.
        is_approved: True, if the requested training plan is one of the approved training plan by the node
        msg: Message from node based on state of the reply
        training_plan_url: The training plan that has been checked for approval
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed

    """

    researcher_id: str
    node_id: str
    job_id: str
    success: bool
    approval_obligation: bool
    status: str
    msg: str
    training_plan_url: str
    command: str


# Ping messages

@catch_dataclass_exception
@dataclass
class PingRequest(Message, RequiresProtocolVersion):
    """Describes a ping message sent by the researcher

    Attributes:
        researcher_id: Id of the researcher that send ping reqeust
        sequence: Ping sequence
        command: Request command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    sequence: int
    command: str



@catch_dataclass_exception
@dataclass
class PingReply(Message, RequiresProtocolVersion):
    """This class describes a ping message sent by the node.

    Attributes:
        researcher_id: Id of the researcher that will receive the reply
        node_id: Node id that replys the request
        succes: True if the node process the request as expected, false if any exception occurs
        sequence: Ping sequence
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    success: bool
    sequence: int
    command: str


# Search messages

@catch_dataclass_exception
@dataclass
class SearchRequest(Message, RequiresProtocolVersion):
    """Describes a search message sent by the researcher.

    Attributes:
        researcher_id: ID of the researcher that sends the request
        tags: Tags for search request
        command: Request command string

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    tags: list
    command: str


@catch_dataclass_exception
@dataclass
class SearchReply(Message, RequiresProtocolVersion):
    """Describes a search message sent by the node

    Attributes:
        researcher_id: Id of the researcher that sends the request
        succes: True if the node process the request as expected, false if any exception occurs
        databases: List of datasets
        node_id: Node id that replys the request
        count: Number of datasets
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    success: bool
    databases: list
    node_id: str
    count: int
    command: str


# Secure aggregation messages

@catch_dataclass_exception
@dataclass
class SecaggDeleteRequest(Message, RequiresProtocolVersion):
    """Describes a secagg context element delete request message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests deletion
        secagg_id: ID of secagg context element that is sent by researcher
        sequence: (unique) sequence number which identifies the message
        element: Type of secagg context element
        job_id: Id of the Job to which this secagg context element is attached
        command: Request command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    secagg_id: str
    sequence: int
    element: int
    job_id: (str, type(None))
    command: str

@catch_dataclass_exception
@dataclass
class SecaggDeleteReply(Message, RequiresProtocolVersion):
    """Describes a secagg context element delete reply message sent by the node

    Attributes:
        researcher_id: ID of the researcher that requests deletion
        secagg_id: ID of secagg context element that is sent by researcher
        sequence: (unique) sequence number which identifies the message
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replies to the request
        msg: Custom message
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    secagg_id: str
    sequence: int
    success: bool
    node_id: str
    msg: str
    command: str

@catch_dataclass_exception
@dataclass
class SecaggRequest(Message, RequiresProtocolVersion):
    """Describes a secagg context element setup request message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests setup
        secagg_id: ID of secagg context element that is sent by researcher
        sequence: (unique) sequence number which identifies the message
        element: Type of secagg context element
        job_id: Id of the Job to which this secagg context element is attached
        parties: List of parties participating to the secagg context element setup
        command: Request command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    secagg_id: str
    sequence: int
    element: int
    job_id: (str, type(None))
    parties: list
    command: str

@catch_dataclass_exception
@dataclass
class SecaggReply(Message, RequiresProtocolVersion):
    """Describes a secagg context element setup reply message sent by the node

    Attributes:
        researcher_id: ID of the researcher that requests setup
        secagg_id: ID of secagg context element that is sent by researcher
        sequence: (unique) sequence number which identifies the message
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replies to the request
        msg: Custom message
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    secagg_id: str
    sequence: int
    success: bool
    node_id: str
    msg: str
    command: str

# Train messages

@catch_dataclass_exception
@dataclass
class TrainRequest(Message, RequiresProtocolVersion):
    """Describes a train message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests training
        job_id: Id of the Job that is sent by researcher
        params_url: URL where model parameters are uploaded
        training_args: Arguments for training routine
        dataset_id: id of the dataset that is used for training
        training: Declares whether training will be performed
        model_args: Arguments to initialize training plan class
        training_plan_url: URL where TrainingPlan is available
        training_plan_class: Class name of the training plan
        command: Reply command string
        aggregator_args: ??
        aux_var_urls: Optional list of URLs where Optimizer auxiliary
            variables files are available

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    job_id: str
    params_url: str
    training_args: dict
    dataset_id: str
    training: bool
    model_args: dict
    training_plan_url: str
    training_plan_class: str
    command: str
    secagg_servkey_id: (str, type(None))
    secagg_biprime_id: (str, type(None))
    secagg_random: (float, type(None))
    secagg_clipping_range: (int, type(None))
    round: int
    aggregator_args: dict
    aux_var_urls: (list, type(None))


@catch_dataclass_exception
@dataclass
class TrainReply(Message, RequiresProtocolVersion):
    """Describes a train message sent by the node.

    Attributes:
        researcher_id: Id of the researcher that receives the reply
        job_id: Id of the Job that is sent by researcher
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replys the request
        dataset_id: id of the dataset that is used for training
        params_url: URL of parameters uploaded by node
        timing: Timing statistics
        msg: Custom message
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    job_id: str
    success: bool
    node_id: str
    dataset_id: str
    params_url: str
    timing: dict
    sample_size: (int, type(None))
    msg: str
    command: str


class MessageFactory:
    """Pack message contents into the appropriate Message class."""
    @staticmethod
    def _raise_for_missing_command(params: Dict[str, Any]):
        """Raise FedBiomedMessageError if input does not contain the `command` key"""
        if "command" not in params:
            _msg = ErrorNumbers.FB601.value + ": message type not specified"
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

    @staticmethod
    def _validate_message_type_or_raise(message_type: str, type_map: Dict[str, Message]):
        """Raise FedbiomedMessageError if `message_tpe` not in `type_map`"""
        if message_type not in type_map:
            _msg = ErrorNumbers.FB601.value + ": bad message type for format_incoming_message: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

    @classmethod
    def format_incoming_message(cls, params: Dict[str, Any]) -> Message:
        """Format a dictionary representing an incoming message into the appropriate Message class.

        Packs the input into the appropriate Message class representing an incoming message.
        The type of Message class is inferred from the `command` key in the input dictionary.
        This function also validates:

        - the legacy of the message
        - the structure of the received message

        Attributes:
            params: the dictionary of key-value pairs extracted from the received message.

        Raises:
            FedbiomedMessageError: if 'command' field is not present in `params`
            FedbiomedMessageError: if the component is not allowed to receive the message

        Returns:
            The received message formatted as an instance of the appropriate Message class
        """
        MessageFactory._raise_for_missing_command(params)
        message_type = params['command']
        MessageFactory._validate_message_type_or_raise(message_type, cls.INCOMING_MESSAGE_TYPE_TO_CLASS_MAP)
        return cls.INCOMING_MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)

    @classmethod
    def format_outgoing_message(cls, params: Dict[str, Any]) -> Message:
        """Format a dictionary representing an outgoing message into the appropriate Message class.

        Packs the input into the appropriate Message class representing an outbound message.
        The type of Message class is inferred from the `command` key in the input dictionary.
        This function also validates:

        - the legacy of the message
        - the structure of the received message

        Attributes:
            params: the dictionary of key-value pairs to be packed into the outgoing message.

        Raises:
            FedbiomedMessageError: if 'command' field is not present in `params`
            FedbiomedMessageError: if the component is not allowed to send the message

        Returns:
            The outbound message formatted as an instance of the appropriate Message class
        """

        MessageFactory._raise_for_missing_command(params)
        message_type = params['command']
        MessageFactory._validate_message_type_or_raise(message_type, cls.OUTGOING_MESSAGE_TYPE_TO_CLASS_MAP)
        params['protocol_version'] = str(__messaging_protocol_version__)  # inject procotol version only in outgoing msg
        return cls.OUTGOING_MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)


class ResearcherMessages(MessageFactory):
    """Specializes MessageFactory for Researcher.

    Researchers send requests and receive replies.
    """
    INCOMING_MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainReply,
                                          'search': SearchReply,
                                          'pong': PingReply,
                                          'log': LogMessage,
                                          'error': ErrorMessage,
                                          'list': ListReply,
                                          'add_scalar': AddScalarReply,
                                          'training-plan-status': TrainingPlanStatusReply,
                                          'approval': ApprovalReply,
                                          'secagg': SecaggReply,
                                          'secagg-delete': SecaggDeleteReply
                                          }

    OUTGOING_MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainRequest,
                                          'search': SearchRequest,
                                          'ping': PingRequest,
                                          'list': ListRequest,
                                          'training-plan-status': TrainingPlanStatusRequest,
                                          'approval': ApprovalRequest,
                                          'secagg': SecaggRequest,
                                          'secagg-delete': SecaggDeleteRequest
                                          }


class NodeMessages(MessageFactory):
    """Specializes MessageFactory for Node.

    Node send replies and receive requests.
    """
    INCOMING_MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainRequest,
                                          'search': SearchRequest,
                                          'ping': PingRequest,
                                          'list': ListRequest,
                                          'training-plan-status': TrainingPlanStatusRequest,
                                          'approval': ApprovalRequest,
                                          'secagg': SecaggRequest,
                                          'secagg-delete': SecaggDeleteRequest
                                          }

    OUTGOING_MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainReply,
                                          'search': SearchReply,
                                          'pong': PingReply,
                                          'log': LogMessage,
                                          'error': ErrorMessage,
                                          'add_scalar': AddScalarReply,
                                          'list': ListReply,
                                          'training-plan-status': TrainingPlanStatusReply,
                                          'approval': ApprovalReply,
                                          'secagg': SecaggReply,
                                          'secagg-delete': SecaggDeleteReply
                                          }
