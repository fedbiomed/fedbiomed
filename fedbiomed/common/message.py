'''
Definition of messages exchanged by the researcher and the nodes
'''

from dataclasses import dataclass
from typing import Dict, Any, Union

from fedbiomed.common.constants  import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.common.logger     import logger


def catch_dataclass_exception(initial_class):
    """
    Decorator: it encapsulate the __init__() method of dataclass
    in order to transform the exceptions sent by the dataclass
    into oour own exception (FedbiomedMessageError)
    """
    class NewCls():
        """
        Class container to wrap the old class into a decorated class
        """
        def __init__(self, *args, **kwargs):
            #
            try:
                self.initial_instance = initial_class(*args, **kwargs)
            except TypeError as e:
                # this is the error raised by dataclass if number of parameter is wrong
                _msg = ErrorNumbers.FB601.value + ": bad number of parameters: " + str(e)
                logger.error(_msg)
                raise FedbiomedMessageError(_msg)

        def __getattribute__(self, s):
            """
            this is called whenever any attribute of a NewCls object is accessed.
            This function first tries to get the attribute of NewCls and run it

            if it fails, it then call the attributes of the initial class
            """
            try:
                _x = super().__getattribute__(s)
            except AttributeError:
                _x = self.initial_instance.__getattribute__(s)
                return _x
            else:
                return _x

    return NewCls


class Message(object):
    """
    This class is a top class for all fedbiomed messages providing all methods
    to access the messaeges

    The subclasses of this class will be pure data containers (no provided functions)
    """

    def __post_init__(self):
        """
        raise FedbiomedMessageError (FB601 error) if parameters of bad type
        remark: this is not check by @dataclass
        """
        if not self.__validate(self.__dataclass_fields__.items()):
            _msg = ErrorNumbers.FB601.value + ": bad input value for message: " + self.__str__()
            logger.critical(_msg)
            raise FedbiomedMessageError(_msg)

    def get_param(self, param: str):
        """This method allows to get the value of a given param

        Args:
            param (str): name of the param
        """
        return(getattr(self, param))

    def get_dict(self) -> Dict[str, Any]:
        """Returns pairs (Message class attributes name, attributes values)
        into a dictionary

        """
        return(self.__dict__)

    def __validate(self, fields: Dict[str, Any]) -> bool:
        """checks whether incoming field types match with attributes
            class type.

        Args:
            fields (Dict[str, Any]): incoming fields

        Returns:
            bool: If validated, ie everything matches,
            returns True, else returns False.
        """
        ret = True
        for field_name, field_def in fields:
            actual_type = type(getattr(self, field_name))
            if actual_type != field_def.type:
                logger.critical(f"{field_name}: '{actual_type}' instead of '{field_def.type}'")
                ret = False
        return ret


@catch_dataclass_exception
@dataclass
class ModelStatusReply(Message):

    """This class describes a model approve status check message sent
       by the node

    Args:
        Message ([type]): Parent class allows to get and set message params

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed

    Keys:
        researcher_id       : Id of the researcher that sends the request
        node_id             : Node id that replys the request
        job_id              : job id related to the experiment
        succes              : True if the node process the request as expected, false
                              if any execption occurs
        approval_obligation : Approval mode for node. True, if model approval is enabled/required
                              in the node for training.
        is_approved         : True, if the requested model is one of the approved model by the node
        msg                 : Message from node based on state of the reply
        model_url           : The model that has been checked for approval
        command             : Reply command
    """

    researcher_id: str
    node_id: str
    job_id: str
    success : bool
    approval_obligation : bool
    is_approved : bool
    msg: str
    model_url: str
    command: str


@catch_dataclass_exception
@dataclass
class SearchReply(Message):
    """This class describes a search message sent by the node

    Args:
        Message ([type]): Parent class allows to get and set message params

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    success: bool
    databases: list
    count: int
    node_id: str
    command: str


@catch_dataclass_exception
@dataclass
class ListReply(Message):

    """This class describes a list reply message sent by the node that includes
    list of datasets. It is a reply for ListRequest messsage from the researcher.

    Args:
        Message ([type]): Parent class allows to get and set message params

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    success: bool
    databases: list
    node_id: str
    command: str
    count: int



@catch_dataclass_exception
@dataclass
class PingReply(Message):
    """
    This class describes a ping message sent by the node

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    success: bool
    sequence: int
    command: str



@catch_dataclass_exception
@dataclass
class TrainReply(Message):
    """
    This class describes a train message sent by the node

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
    msg: str
    command: str


@catch_dataclass_exception
@dataclass
class AddScalarReply(Message):
    """
    This class describes a add_scalar message sent by the node

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    job_id: str
    key: str
    value: float
    epoch: int
    total_samples: int
    batch_samples: int
    num_batches: int
    result_for: str
    iteration: int
    command: str



@catch_dataclass_exception
@dataclass
class LogMessage(Message):
    """
    This class describes a log message sent by the node

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    level: str
    msg: str
    command: str


@catch_dataclass_exception
@dataclass
class ErrorMessage(Message):
    """
    This class describes an error message sent by the node

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    node_id: str
    errnum: ErrorNumbers
    extra_msg: str
    command: str



@catch_dataclass_exception
@dataclass
class ModelStatusRequest(Message):

    """This class describes a model approve status check message sent
       by the researcher

    Args:
        Message ([type]): Parent class allows to get and set message params

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed


    Keys:
        researcher_id   : Id of the researcher that sends the request
        job_id          : job id related to the experiment.
        model_url       : The model that is going to be checked for approval
        command         : Request command
    """

    researcher_id: str
    job_id: str
    model_url: str
    command: str



@catch_dataclass_exception
@dataclass
class SearchRequest(Message):
    """
    This class describes a search message sent by the researcher

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    tags: list
    command: str



@catch_dataclass_exception
@dataclass
class ListRequest(Message):
    """
    This class describes a list request message sent by the researcher to nodes in order to list
    datasets belonging to each node.

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    command: str


@catch_dataclass_exception
@dataclass
class PingRequest(Message):
    """
    This class describes a ping message sent by the researcher

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    sequence: int
    command: str



@catch_dataclass_exception
@dataclass
class TrainRequest(Message):
    """
    This class describes a train message sent by the researcher

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    researcher_id: str
    job_id: str
    params_url: str
    training_args: dict
    training_data: dict
    model_args: dict
    model_url: str
    model_class: str
    command: str



class ResearcherMessages():
    """This class allows to create the corresponding class instance from
    a received/ sent message by the researcher
    """
    @classmethod
    def reply_create(cls, params: Dict[str, Any]) -> Union[TrainReply,
                                                           SearchReply,
                                                           PingReply,
                                                           LogMessage,
                                                           ErrorMessage,
                                                           ListReply,
                                                           AddScalarReply,
                                                           ModelStatusReply]:
        """this method is used on message reception (as a mean to reply to
        node requests, such as a Ping request).
        it creates the adequate message, it maps an instruction
        (given the key "command" in the input dictionary `params`)
        to a Message object
        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
        FedbiomedMessageError: triggered if the message is not allowed to
        be received by the researcher
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
                                     'model-status': ModelStatusReply
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
                                                             ModelStatusRequest]:

        """This method creates the adequate message/request,
        it maps an instruction (given the key "command" in
        the input dictionary `params`) to a Message object

        It validates:
        - the legagy of the message
        - the structure of the created message

        Args:
        params (dict): dictionary containing the message.

        Raises:
            FedbiomedMessageError: if the message is not allowed to be sent by the researcher
            KeyError ?
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
                                     'model-status': ModelStatusRequest
                                     }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for request_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)


class NodeMessages():
    """This class allows to create the corresponding class instance from
    a received/sent message by the Node
    """
    @classmethod
    def request_create(cls, params: dict) -> Union[TrainRequest,
                                                   SearchRequest,
                                                   PingRequest,
                                                   ListRequest,
                                                   ModelStatusRequest]:
        """
        This method creates the adequate message/ request to send
        to researcher, it maps an instruction (given the key "command" in the
        input dictionary `params`) to a Message object

        It validates:
        - the legagy of the message
        - the structure of the created message

        Raises:
            FedbiomedMessageError: triggered if the message is not allowed te be sent
            by the node (ie if message `command` field is not either a
            train request, search request or a ping request)

        Returns:
            An instance of the corresponding class (TrainRequest,
            SearchRequest, PingRequest)
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
                                     'model-status': ModelStatusRequest
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
                                                 ModelStatusReply]:
        """this method is used on message reception.
        It creates the adequate message reply to send to the researcher,
        it maps an instruction (given the key "command" in the
        input dictionary `params`) to a Message object
        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
            FedbiomedMessageError: if the message is not allowed te be received by
            the node (ie if message `command` field is not either a
            train request, search request, a ping request, add scalar
            request, or error message)

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
                                     'model-status': ModelStatusReply
                                     }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            _msg = ErrorNumbers.FB601.value + ": bad message type for request_create: {}".format(message_type)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg)
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)
