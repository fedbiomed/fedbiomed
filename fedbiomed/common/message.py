# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Definition of messages exchanged by the researcher and the nodes
"""

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, get_args

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtobufMessage

import fedbiomed.transport.protocols.researcher_pb2 as r_pb2
from fedbiomed.common.constants import ErrorNumbers, __messaging_protocol_version__
from fedbiomed.common.exceptions import FedbiomedMessageError, FedbiomedValueError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import import_object, raise_for_version_compatibility


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
            self.__class__.__dict__["__initial_init__"](self, *args, **kwargs)

        except TypeError as e:
            # this is the error raised by dataclass if number of parameter is wrong
            _msg = ErrorNumbers.FB601.value + ": bad number of parameters: " + str(e)
            logger.error(_msg)
            raise FedbiomedMessageError(_msg) from e

    @functools.wraps(cls)
    def wrap(cls: Callable):
        """Wrapper to the class given as parameter

        Class wrapping should keep some attributes (__doc__, etc) of the initial class
        or the API documentation tools will be mistaken

        """
        cls.__initial_init__ = cls.__init__
        setattr(cls, "__init__", __cde_init__)

        return cls

    return wrap(cls)


class Message:
    """Base class for all fedbiomed messages providing all methods
    to access the messages

    The subclass of this class will be pure data containers (no provided functions)
    """

    def __post_init__(self):
        """Post init of dataclass

        - remark: this is not check by @dataclass

        Raises:
            FedbiomedMessageError: (FB601 error) if parameters of bad type

        """

        if not self.__validate(self.__dataclass_fields__.items()):
            _msg = (
                ErrorNumbers.FB601.value
                + ": bad input value for message: "
                + self.__str__()[0:1000]
                + "..."
            )
            logger.critical(_msg)
            raise FedbiomedMessageError(_msg)

    @property
    def __name__(self) -> str:
        """Property for message name

        Returns:
            Message type name
        """
        return type(self).__name__

    def get_param(self, param: str):
        """Get the value of a given param

        Args:
            param: name of the param
        """
        return getattr(self, param, None)

    def get_dict(self) -> Dict[str, Any]:
        """Returns pairs (Message class attributes name, attributes values) into a dictionary

        Returns:
            Message as dictionary
        """
        return {**self.__dict__}

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
                logger.critical(
                    f"{field_name}: '{value}' instead of '{field_def.type}'"
                )
                ret = False
        return ret

    def to_dict(self) -> Dict:
        """Serializes the message

        Returns:
            Serializes data class into a dict
        """
        class_ = type(self).__name__
        module_ = type(self).__module__

        return {
            **self.get_dict(),
            "__type_message__": {"module": module_, "class": class_},
        }

    @staticmethod
    def from_dict(obj: Dict):
        """De-serializes the message"""

        message = {**obj}

        if (
            "__type_message__" not in message
            or not isinstance(message["__type_message__"], dict)
            or not all(
                x_ in message["__type_message__"].keys() for x_ in ["class", "module"]
            )
        ):
            raise FedbiomedValueError(
                "Message does not include valid '__type_message__' entry."
            )

        type_ = message["__type_message__"]
        cls_ = import_object("fedbiomed.common.message", type_["class"])

        if not inspect.isclass(cls_) or not issubclass(cls_, Message):
            raise FedbiomedMessageError(
                "Given object class is not subclass of 'Message'"
            )

        raise_for_version_compatibility(
            message["protocol_version"], __messaging_protocol_version__
        )

        message.pop("__type_message__")

        return cls_(**message)

    def to_proto(self):
        """Converts recursively python dataclass to gRPC proto"""

        proto_dict = {}
        for key, _ in self.__dataclass_fields__.items():
            param = self.get_param(key)
            if hasattr(param, "__PROTO_TYPE__"):
                proto_dict.update({key: self.get_param(key).to_proto()})
            else:
                proto_dict.update({key: self.get_param(key)})

        return self.__PROTO_TYPE__(**proto_dict)

    @classmethod
    def from_proto(cls, proto: ProtobufMessage) -> Dict[str, Any]:
        """Converts given protobuf to python Dict"""

        dict_ = {}
        one_ofs = proto.DESCRIPTOR.oneofs_by_name
        for field in proto.DESCRIPTOR.fields:

            one_of_field = False
            for one_of, _ in one_ofs.items():
                if field.name == proto.WhichOneof(one_of):
                    one_of_field = True

            # If the field is oneof and options are in message type
            if one_of_field and field.type == FieldDescriptor.TYPE_MESSAGE:

                field_ = cls.__dataclass_fields__[field.name]
                args = get_args(field_.type)

                # Make sure oneof message is typed as Optional
                if not args:
                    raise FedbiomedMessageError(
                        f"Please make sure the field '{field_.name}' in dataclass '{cls.__name__}' "
                        "is typed as Optional[<dataclass>]. The field that are typed as `oneof` "
                        "in proto file should be typed as Optional in python dataclass"
                    )

                if not hasattr(args[0], "__PROTO_TYPE__"):
                    raise FedbiomedMessageError(
                        f"Dataclass {args[0]} should have attribute '__PROTO_TYPE__'"
                    )

                dict_.update(
                    {field.name: args[0].from_proto(getattr(proto, field.name))}
                )

            # Detects the types that are declared as `optional`
            # NOTE: In proto3 all fields are labeled as `LABEL_OPTIONAL` by default.
            # However, if the field is labeled as `optional` explicitly, it will have
            # presence, otherwise, `has_presence` returns False
            elif field.has_presence and field.label == FieldDescriptor.LABEL_OPTIONAL:

                # If proto has the field it means that the value is not None
                if proto.HasField(field.name):
                    dict_.update({field.name: getattr(proto, field.name)})

            elif field.label == FieldDescriptor.LABEL_REPEATED:

                if field.type == FieldDescriptor.TYPE_MESSAGE:
                    dict_.update({field.name: dict(getattr(proto, field.name))})
                else:
                    dict_.update({field.name: list(getattr(proto, field.name))})

            else:
                dict_.update({field.name: getattr(proto, field.name)})

        return cls(**dict_)


#
# messages definition, sorted by
# - alphabetic order
# - Request/Reply regrouping
#


@dataclass
class ProtoSerializableMessage(Message):
    pass


@dataclass(kw_only=True)
class RequestReply(Message):
    """Common attribute for Request and Reply Message.

    Attributes:
        request_id: unique ID for this request-reply
    """

    request_id: Optional[str] = None


@dataclass(kw_only=True)
class RequiresProtocolVersion:
    """Mixin class for messages that must be endowed with a version field.

    Attributes:
        protocol_version: version of the messaging protocol used
    """

    # Adds default protocol version thanks to `kw_oly  True`
    protocol_version: str = str(__messaging_protocol_version__)


@dataclass(kw_only=True)
class OverlayMessage(Message, RequiresProtocolVersion):
    """Message for handling overlay trafic.

    Same message used from source node to researcher, and from researcher to destination node.

    Attributes:
        researcher_id: Id of the researcher relaying the overlay message
        node_id: Id of the source node of the overlay message
        dest_node_id: Id of the destination node of the overlay message
        overlay: payload of the message to be forwarded unchanged to the destination node
        setup: True if this is a channel setup message, False if this is an application layer message
        salt: value used for salting the key derivation for this message
        nonce: value used for noncing the encryption for this message

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str  # Needed for source and destination node side message handling
    node_id: str  # Needed for researcher side message handling (receiving a `ReplyTask`)
    dest_node_id: str  # Needed for researcher side message handling
    overlay: bytes
    setup: bool
    salt: bytes
    nonce: bytes


@dataclass(kw_only=True)
class InnerMessage(Message):
    """Parent class of messages sent from node to node.

    Node to node messages are sent as inner message (payload) of an overlay message

    Attributes:
        node_id: Id of the source node sending the mess
        dest_node_id: Id of the destination node of the overlay message
    """

    # Needed by destination node for easily identifying source node.
    # Not needed for security if message is securely signed by source node.
    node_id: str
    # Needed for security if we `encrypt(sign(message))` to link signed message to identity of destination node
    # and prevent replay of message by a malicious node to another node
    # https://theworld.com/~dtd/sign_encrypt/sign_encrypt7.html
    dest_node_id: str

    # caveat: InnerMessage (without `request_id`) leaves room for replay attacks


@dataclass(kw_only=True)
class InnerRequestReply(InnerMessage):
    """Common attribute for Request and Reply Inner Message.

    Attributes:
        request_id: unique ID for this request-reply
    """

    request_id: Optional[str] = None


# --- gRPC messages --------------------------------------------------------------------------------


@dataclass
class Log(ProtoSerializableMessage):
    """Describes the message type for log coming from node to researcher"""

    __PROTO_TYPE__ = r_pb2.FeedbackMessage.Log

    node_id: str
    level: str
    msg: str


@catch_dataclass_exception
@dataclass
class Scalar(ProtoSerializableMessage):
    """Describes a add_scalar message sent by the node.

    Attributes:
        researcher_id: ID of the researcher that receives the reply
        experiment_id: ID of the experiment that is sent by researcher
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

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed

    """

    __PROTO_TYPE__ = r_pb2.FeedbackMessage.Scalar

    node_id: str
    experiment_id: str
    train: bool
    test: bool
    test_on_local_updates: bool
    test_on_global_updates: bool
    metric: dict
    total_samples: int
    batch_samples: int
    num_batches: int
    iteration: int
    epoch: Optional[int] = None
    num_samples_trained: Optional[int] = None


@dataclass
class TaskRequest(ProtoSerializableMessage, RequiresProtocolVersion):
    """Task request message from node to researcher"""

    __PROTO_TYPE__ = r_pb2.TaskRequest
    node: str


@dataclass
class TaskResponse(ProtoSerializableMessage):
    """Response for task request"""

    __PROTO_TYPE__ = r_pb2.TaskResponse

    size: int
    iteration: int
    bytes_: bytes


@dataclass
class TaskResult(ProtoSerializableMessage):
    """Response for task request"""

    __PROTO_TYPE__ = r_pb2.TaskResult

    size: int
    iteration: int
    bytes_: bytes


@dataclass
class FeedbackMessage(ProtoSerializableMessage, RequiresProtocolVersion):
    __PROTO_TYPE__ = r_pb2.FeedbackMessage

    researcher_id: Optional[str] = None
    log: Optional[Log] = None
    scalar: Optional[Scalar] = None


# --- Node <=> Node messages ----------------------------------------------------

@catch_dataclass_exception
@dataclass
class ChannelSetupRequest(InnerRequestReply, RequiresProtocolVersion):
    """Message for requesting peer node key for securing a n2n channel.

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """


@catch_dataclass_exception
@dataclass
class ChannelSetupReply(InnerRequestReply, RequiresProtocolVersion):
    """Message for reply peer node key for securing a n2n channel.

    Attributes:
        public_key: public key of replying node

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """
    public_key: bytes


@catch_dataclass_exception
@dataclass
class KeyRequest(InnerRequestReply, RequiresProtocolVersion):
    """Message for starting a new exchange for creating crypto key material.

    Currently only Diffie-Hellman key exchange is supported

    Attributes:
        secagg_id: unique ID of this secagg context element

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    secagg_id: str


@catch_dataclass_exception
@dataclass
class KeyReply(InnerRequestReply, RequiresProtocolVersion):
    """Message for continuing an exchange for creating crypto key material.

    Currently only Diffie-Hellman key exchange is supported

    Attributes:
        public_key: public key of replying node
        secagg_id: unique ID of this secagg context element

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    public_key: bytes
    secagg_id: str


@catch_dataclass_exception
@dataclass
class AdditiveSSharingRequest(InnerRequestReply, RequiresProtocolVersion):
    secagg_id: str


@catch_dataclass_exception
@dataclass
class AdditiveSSharingReply(InnerRequestReply, RequiresProtocolVersion):
    secagg_id: str
    share: list | int


# --- Node <=> Researcher messages ----------------------------------------------


@catch_dataclass_exception
@dataclass
class ApprovalRequest(RequestReply, RequiresProtocolVersion):
    """Describes the TrainingPlan approval request from researcher to node.

    Attributes:
        researcher_id: id of the researcher that sends the request
        description: description of the training plan
        training_plan: The training plan that is going to be checked for approval

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    description: str
    training_plan: str


@catch_dataclass_exception
@dataclass
class ApprovalReply(RequestReply, RequiresProtocolVersion):
    """Describes the TrainingPlan approval reply (acknoledge) from node to researcher.

    Attributes:
        researcher_id: Id of the researcher that will receive the reply
        training_plan_id: Unique training plan identifier, can be none in case of
            success false.
        message: currently unused (empty string)
        node_id: Node id that replys the request
        status: status code for the request (obsolete, always 0)
        success: Request was successfully sumbitted to node (not yet approved)

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    training_plan_id: str | None
    message: str
    node_id: str
    status: int
    success: bool


# Error message


@catch_dataclass_exception
@dataclass(kw_only=True)
class ErrorMessage(RequestReply, RequiresProtocolVersion):
    """Describes an error message sent by the node.

    Attributes:
        researcher_id: ID of the researcher that receives the error message
        node_id: ID of the node that sends error message
        errnum: Error ID/Number
        extra_msg: Additional message regarding the error

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    node_id: str
    extra_msg: str
    errnum: Optional[str] = None


@catch_dataclass_exception
@dataclass
class ListRequest(RequestReply, RequiresProtocolVersion):
    """Describes a list request message sent by the researcher to nodes in order to list datasets belonging to
    each node.

    Attributes:
        researcher_id: Id of the researcher that sends the request

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str


@catch_dataclass_exception
@dataclass
class ListReply(RequestReply, RequiresProtocolVersion):
    """This class describes a list reply message sent by the node that includes list of datasets. It is a
    reply for ListRequest message from the researcher.

    Attributes:
        researcher_id: Id of the researcher that sends the request
        succes: True if the node process the request as expected, false if any exception occurs
        databases: List of datasets
        node_id: Node id that replys the request
        count: Number of datasets

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    success: bool
    databases: list
    node_id: str
    count: int


# Ping messages


@catch_dataclass_exception
@dataclass
class PingRequest(RequestReply, RequiresProtocolVersion):
    """Describes a ping message sent by the researcher

    Attributes:
        researcher_id: Id of the researcher that send ping request

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str


@catch_dataclass_exception
@dataclass
class PingReply(RequestReply, RequiresProtocolVersion):
    """This class describes a ping message sent by the node.

    Attributes:
        researcher_id: Id of the researcher that will receive the reply
        node_id: Node id that replys the request
        succes: True if the node process the request as expected, false if any exception occurs

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    node_id: str


# Search messages


@catch_dataclass_exception
@dataclass
class SearchRequest(RequestReply, RequiresProtocolVersion):
    """Describes a search message sent by the researcher.

    Attributes:
        researcher_id: ID of the researcher that sends the request
        tags: Tags for search request

    Raises:
       FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    tags: list


@catch_dataclass_exception
@dataclass
class SearchReply(RequestReply, RequiresProtocolVersion):
    """Describes a search message sent by the node

    Attributes:
        researcher_id: Id of the researcher that sends the request
        succes: True if the node process the request as expected, false if any exception occurs
        databases: List of datasets
        node_id: Node id that replys the request
        count: Number of datasets

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    databases: list
    node_id: str
    count: int


# Secure aggregation messages


@catch_dataclass_exception
@dataclass
class SecaggDeleteRequest(RequestReply, RequiresProtocolVersion):
    """Describes a secagg context element delete request message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests deletion
        secagg_id: ID of secagg context element that is sent by researcher
        element: Type of secagg context element
        experiment_id: Id of the experiment to which this secagg context element is attached

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    secagg_id: str
    element: int
    experiment_id: Optional[str]


@catch_dataclass_exception
@dataclass(kw_only=True)
class SecaggDeleteReply(RequestReply, RequiresProtocolVersion):
    """Describes a secagg context element delete reply message sent by the node

    Attributes:
        researcher_id: ID of the researcher that requests deletion
        secagg_id: ID of secagg context element that is sent by researcher
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replies to the request
        msg: Custom message

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    secagg_id: str
    success: bool
    node_id: str
    msg: Optional[str] = None


@catch_dataclass_exception
@dataclass(kw_only=True)
class SecaggRequest(RequestReply, RequiresProtocolVersion):
    """Describes a secagg context element setup request message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests setup
        secagg_id: ID of secagg context element that is sent by researcher
        element: Type of secagg context element
        experiment_id: Id of the experiment to which this secagg context element is attached
        parties: List of parties participating to the secagg context element setup

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    secagg_id: str
    element: int
    experiment_id: str
    parties: list


@catch_dataclass_exception
@dataclass(kw_only=True)
class SecaggReply(RequestReply, RequiresProtocolVersion):
    """Describes a secagg context element setup reply message sent by the node

    Attributes:
        researcher_id: ID of the researcher that requests setup
        secagg_id: ID of secagg context element that is sent by researcher
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replies to the request
        msg: Custom message

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    secagg_id: str
    success: bool
    node_id: str
    msg: Optional[str] = None
    msg: str


@catch_dataclass_exception
@dataclass
class AdditiveSSSetupRequest(SecaggRequest):
    """Message to request secure aggregation setup from researcher to nodes"""


@catch_dataclass_exception
@dataclass(kw_only=True)
class AdditiveSSSetupReply(SecaggReply):
    """Message that instantiated on the node side to reply secagg setup request from researcher"""

    share: int | list


@catch_dataclass_exception
@dataclass
class TrainingPlanStatusRequest(RequestReply, RequiresProtocolVersion):
    """Describes a training plan approve status check message sent by the researcher.

    Attributes:
        researcher_id: Id of the researcher that sends the request
        experiment_id: experiment id related to the experiment.
        training_plan_url: The training plan that is going to be checked for approval

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    experiment_id: str
    training_plan: str


@catch_dataclass_exception
@dataclass
class TrainingPlanStatusReply(RequestReply, RequiresProtocolVersion):
    """Describes a training plan approve status check message sent by the node

    Attributes:
        researcher_id: Id of the researcher that sends the request
        node_id: Node id that replies the request
        experiment_id: experiment id related to the experiment
        success: True if the node process the request as expected, false
            if any exception occurs
        approval_obligation : Approval mode for node. True, if
            training plan approval is enabled/required in the node for training.
        status: a `TrainingPlanApprovalStatus` value describing the approval status
        msg: Message from node based on state of the reply
        training_plan: The training plan that has been checked for approval
        training_plan_id: Unique training plan identifier, can be None in case of
            success false.

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed

    """

    researcher_id: str
    node_id: str
    experiment_id: str
    success: bool
    approval_obligation: bool
    status: str
    msg: str
    training_plan: str
    training_plan_id: Optional[str]


# Train messages


@catch_dataclass_exception
@dataclass
class TrainRequest(RequestReply, RequiresProtocolVersion):
    """Describes a train message sent by the researcher

    Attributes:
        researcher_id: ID of the researcher that requests training
        experiment_id: Id of the experiment that is sent by researcher
        state_id: ID of state associated to this request on node
        training_args: Arguments for training routine
        dataset_id: id of the dataset that is used for training
        training: Declares whether training will be performed
        model_args: Arguments to initialize training plan class
        training_plan: Source code of training plan
        training_plan_class: Class name of the training plan
        round: number of rounds already executed for this experiment
        aggregator_args: ??
        aux_var: Optimizer auxiliary variables
        optim_aux_var: Optional dict of Optimizer auxiliary variables

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    experiment_id: str
    state_id: Optional[str]
    training_args: dict
    dataset_id: str
    training: bool
    model_args: dict
    params: dict
    training_plan: str
    training_plan_class: str
    round: int
    aggregator_args: Dict
    secagg_arguments: Optional[Dict] = None
    optim_aux_var: Optional[Dict] = None


@catch_dataclass_exception
@dataclass
class TrainReply(RequestReply, RequiresProtocolVersion):
    """Describes a train message sent by the node.

    Attributes:
        researcher_id: Id of the researcher that receives the reply
        experiment_id: Id of the experiment that is sent by researcher
        success: True if the node process the request as expected, false if any exception occurs
        node_id: Node id that replies the request
        dataset_id: id of the dataset that is used for training
        params_url: URL of parameters uploaded by node
        timing: Timing statistics
        msg: Custom message

    Raises:
        FedbiomedMessageError: triggered if message's fields validation failed
    """

    researcher_id: str
    experiment_id: str
    success: bool
    node_id: str
    dataset_id: str
    timing: dict
    msg: str
    state_id: Optional[str] = None
    sample_size: Optional[int] = None
    encrypted: bool = False
    params: Optional[Union[Dict, List]] = None  # None for testing only
    optimizer_args: Optional[Dict] = None  # None for testing only
    optim_aux_var: Optional[Dict] = None  # None for testing only
    encryption_factor: Optional[List] = None  # None for testing only
