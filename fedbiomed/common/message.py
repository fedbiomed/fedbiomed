# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
Definition of messages exchanged by the researcher and the nodes
'''

import functools

from dataclasses import dataclass
from typing import Dict, Any, Union, Callable

from fedbiomed.common.constants import ErrorNumbers
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

@catch_dataclass_exception
@dataclass
class AddScalarReply(Message):
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
class ApprovalRequest(Message):
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
class ApprovalReply(Message):
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
class ErrorMessage(Message):
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
class ListRequest(Message):
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
class ListReply(Message):
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
class LogMessage(Message):
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
class TrainingPlanStatusRequest(Message):
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
class TrainingPlanStatusReply(Message):
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
class PingRequest(Message):
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
class PingReply(Message):
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
class SearchRequest(Message):
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
class SearchReply(Message):
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
class SecaggDeleteRequest(Message):
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
    job_id: str
    command: str

@catch_dataclass_exception
@dataclass
class SecaggDeleteReply(Message):
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
class SecaggRequest(Message):
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
    job_id: str
    parties: list
    command: str

@catch_dataclass_exception
@dataclass
class SecaggReply(Message):
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
class TrainRequest(Message):
    """Describes a train message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests training
        job_id: Id of the Job that is sent by researcher
        params_url: URL where model parameters are uploaded
        training_args: Arguments for training routine
        training_data: Dataset meta-data for training
        training: Declares whether training will be performed
        model_args: Arguments to initialize training plan class
        training_plan_url: URL where TrainingPlan is available
        training_plan_class: Class name of the training plan
        command: Reply command string

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    job_id: str
    params_url: str
    training_args: dict
    training_data: dict
    training: bool
    model_args: dict
    training_plan_url: str
    training_plan_class: str
    command: str
    aggregator_args: dict


@catch_dataclass_exception
@dataclass
class TrainReply(Message):
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


# protocol definition

class ResearcherMessages():
    """Allows to create the corresponding class instance from a received/sent message by the researcher."""

    @classmethod
    def reply_create(cls, params: Dict[str, Any]) -> Union[TrainReply,
                                                           SearchReply,
                                                           PingReply,
                                                           LogMessage,
                                                           ErrorMessage,
                                                           ListReply,
                                                           AddScalarReply,
                                                           TrainingPlanStatusReply,
                                                           ApprovalReply,
                                                           SecaggReply,
                                                           SecaggDeleteReply]:
        """Message reception (as a mean to reply to node requests, such as a Ping request).

        It creates the adequate message, it maps an instruction (given the key "command" in the input dictionary
        `params`) to a Message object

        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
            FedbiomedMessageError: triggered if the message is not allowed to be received by the researcher
            KeyError: triggered if 'command' field is not present in `params`

        Returns:
            An instance of the corresponding Message class
        """
        try:
            message_type = params['command']
        except KeyError:
            _msg = ErrorNumbers.FB601.value + ": message type not specified"
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

        MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainReply,
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

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for reply_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)

    @classmethod
    def request_create(cls, params: Dict[str, Any]) -> Union[TrainRequest,
                                                             SearchRequest,
                                                             PingRequest,
                                                             ListRequest,
                                                             TrainingPlanStatusRequest,
                                                             ApprovalRequest,
                                                             SecaggRequest,
                                                             SecaggDeleteRequest]:

        """Creates the adequate message/request,

        It maps an instruction (given the key "command" in the input dictionary `params`) to a Message object

        It validates:
        - the legacy of the message
        - the structure of the created message

        Args:
            params: dictionary containing the message.

        Raises:
            FedbiomedMessageError: if the message is not allowed to be sent by the researcher
            KeyError: Missing key ub the reqeust

        Returns:
            An instance of the corresponding Message class
        """

        try:
            message_type = params['command']
        except KeyError:
            _msg = ErrorNumbers.FB601.value + ": message type not specified"
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

        MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainRequest,
                                     'search': SearchRequest,
                                     'ping': PingRequest,
                                     'list': ListRequest,
                                     'training-plan-status': TrainingPlanStatusRequest,
                                     'approval': ApprovalRequest,
                                     'secagg': SecaggRequest,
                                     'secagg-delete': SecaggDeleteRequest
                                     }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for request_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)


class NodeMessages():
    """Allows to create the corresponding class instance from a received/sent message by the Node"""

    @classmethod
    def request_create(cls, params: dict) -> Union[TrainRequest,
                                                   SearchRequest,
                                                   PingRequest,
                                                   ListRequest,
                                                   TrainingPlanStatusRequest,
                                                   ApprovalRequest,
                                                   SecaggRequest,
                                                   SecaggDeleteRequest]:
        """Creates the adequate message/ request to send to researcher, it maps an instruction (given the key
        "command" in the input dictionary `params`) to a Message object

        It validates:
        - the legacy of the message
        - the structure of the created message

        Raises:
            FedbiomedMessageError: triggered if the message is not allowed te be sent by the node (ie if message
                `command` field is not either a train request, search request or a ping request)

        Returns:
            An instance of the corresponding class (TrainRequest,SearchRequest, PingRequest)
        """
        try:
            message_type = params['command']
        except KeyError:
            _msg = ErrorNumbers.FB601.value + ": message type not specified"
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

        # mapping message type to an object
        MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainRequest,
                                     'search': SearchRequest,
                                     'ping': PingRequest,
                                     'list': ListRequest,
                                     'training-plan-status': TrainingPlanStatusRequest,
                                     'approval': ApprovalRequest,
                                     'secagg': SecaggRequest,
                                     'secagg-delete': SecaggDeleteRequest
                                     }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for reply_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)

    @classmethod
    def reply_create(cls, params: dict) -> Union[TrainReply,
                                                 SearchReply,
                                                 PingReply,
                                                 LogMessage,
                                                 ErrorMessage,
                                                 AddScalarReply,
                                                 ListReply,
                                                 TrainingPlanStatusReply,
                                                 ApprovalReply,
                                                 SecaggReply,
                                                 SecaggDeleteReply]:
        """Message reception.

        It creates the adequate message reply to send to the researcher, it maps an instruction (given the key
        "command" in the input dictionary `params`) to a Message object

        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
            FedbiomedMessageError: if the message is not allowed te be received by the node (ie if message `command`
                field is not either a train request, search request, a ping request, add scalar request, or
                error message)

        Returns:
            An instance of the corresponding class
        """
        try:
            message_type = params['command']
        except KeyError:
            _msg = ErrorNumbers.FB601.value + ": message type not specified"
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)

        MESSAGE_TYPE_TO_CLASS_MAP = {'train': TrainReply,
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

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for request_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)
