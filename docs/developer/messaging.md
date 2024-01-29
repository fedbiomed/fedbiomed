# RPC Protocol and Messages  

The messages/payloads between Fed-BioMed components are defined and typed on the application and the communication (gRPC) layers. The application layer messages are typed and defined in order to use within the application, and the messages for communication are the protobuff object that is used for sending the data from one end-point to another. After a client or the server receives a protobuf it immediately converts it to corresponding dataclass defined in the application layer. Therefore, new messages introduced by developer should be defined in `fedbiomed.common.message` module and should inherit from `fedbiomed.common.message`. However, while some message types does not require to be modified or created as protocol buffers, some does. For example, application messages such as `TrainRequest`, `TrainReply`, `SerachRequest` are typed generally using single protobuf as `TaskResponse`, but some messages are defined in protocol files explicitly as it is in `fedbiomed.message.module` (e.g `Log` and `Scalar`). Following sections will give the details of the messages used in communication and application layers. 


## RPC protocol buffers  

Researcher component is the RPC server that provides RPC services for the nodes. The services and the corresponding message types for services are defined in `fedbiomed/transport/protocols/researcher.proto`. The service and message instances are generated automatically after compiling the protocol files. It is mandatory to define a message for each service and use corresponding Python instance as input or output value of services implemented in the software. 

### Compiling proto files 

Protocol files should be compiled to be able to create corresponding message and service objects in Python. Compilation can be done by executing `{FEDBIOMED_DIR}/scripts/protoc.py`. The script will generate RPC service and protocol buffer for Python in the directory `{FEDBIOMED_DIR}/fedbiomed/transport/protocols`. 


### Example protocol messages and corresponding data classes in the application layer

In `researcher.proto`, the message `TaskResponse` and `TaskResult` is a general format representing the bytes of messages. These messages are converted to corresponding `Message` dataclass of Fed-BioMed after they are received by node or researcher server (e.g `fedbiomed.common.message.TaskResponse`). `TaskResponse` or `TaskResult` can wrap the message related to tasks as bytes. For example, The `bytes_` field of `TaskResult` can contain bytes of `TrainReply` or `SearchReply`. The reason that these messages are typed as bytes is that they can be big message that needs to be sent as stream. Please see the following train request example to understand the workflow in message serialization;

1. Researcher creates `fedbiomed.common.message.TrainRequest` message to assign train task to the nodes. 
2. The node executes `GetTask` service of researcher.
3. Researcher service takes the `TrainRequest`.
    3.1 serializes as bytes.
    3.2 creates `fedbiomed.common.message.TaskResponse` from bytes 
    3.3 converts `TaskResponse` dataclass to `TaskResponse` protobuf using method `fedbiomed.common.message.TaskResponse.to_proto()`
4. send `TaskResponse` to the node. 


However, in `researhcer.proto` some messages are typed and defined explicitly. For example, the message `TrainRequest` is the input value for the service `GetTaskUnary` that is executed by the nodes. `TrainRequest` ptoobuf has also corresponding message type defined in `fedbiomed.common.message` as `TaskRequest`. Please see the implemented dataclass for `TaskRequest` in `message.py` module. 

```
@dataclass
class TaskRequest(ProtoSerializableMessage, RequiresProtocolVersion):
    """Task request message from node to researcher"""
    __PROTO_TYPE__ = r_pb2.TaskRequest
    node: str
```
The messages that have corresponding protocol buffer in communication layer should define `__PROTO_TYPE__` attribute to acknowledge which protobuf will be used for this message class to be able to send it through RPC call. Thanks to this declaration the message can be converted to proto using `.to_proto()` or protobuf can be converted to python dataclass using `.form_proto(proto)` methods.  