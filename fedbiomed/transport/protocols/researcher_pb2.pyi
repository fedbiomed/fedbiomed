from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ProtocolVersion(_message.Message):
    __slots__ = ("protocol_version",)
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    def __init__(self, protocol_version: _Optional[str] = ...) -> None: ...

class FeedbackMessage(_message.Message):
    __slots__ = ("protocol_version", "researcher_id", "scalar", "log")
    class Scalar(_message.Message):
        __slots__ = ("node_id", "experiment_id", "train", "test", "test_on_local_updates", "test_on_global_updates", "metric", "epoch", "total_samples", "batch_samples", "num_batches", "num_samples_trained", "iteration")
        class MetricEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: float
            def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
        TRAIN_FIELD_NUMBER: _ClassVar[int]
        TEST_FIELD_NUMBER: _ClassVar[int]
        TEST_ON_LOCAL_UPDATES_FIELD_NUMBER: _ClassVar[int]
        TEST_ON_GLOBAL_UPDATES_FIELD_NUMBER: _ClassVar[int]
        METRIC_FIELD_NUMBER: _ClassVar[int]
        EPOCH_FIELD_NUMBER: _ClassVar[int]
        TOTAL_SAMPLES_FIELD_NUMBER: _ClassVar[int]
        BATCH_SAMPLES_FIELD_NUMBER: _ClassVar[int]
        NUM_BATCHES_FIELD_NUMBER: _ClassVar[int]
        NUM_SAMPLES_TRAINED_FIELD_NUMBER: _ClassVar[int]
        ITERATION_FIELD_NUMBER: _ClassVar[int]
        node_id: str
        experiment_id: str
        train: bool
        test: bool
        test_on_local_updates: bool
        test_on_global_updates: bool
        metric: _containers.ScalarMap[str, float]
        epoch: int
        total_samples: int
        batch_samples: int
        num_batches: int
        num_samples_trained: int
        iteration: int
        def __init__(self, node_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., train: bool = ..., test: bool = ..., test_on_local_updates: bool = ..., test_on_global_updates: bool = ..., metric: _Optional[_Mapping[str, float]] = ..., epoch: _Optional[int] = ..., total_samples: _Optional[int] = ..., batch_samples: _Optional[int] = ..., num_batches: _Optional[int] = ..., num_samples_trained: _Optional[int] = ..., iteration: _Optional[int] = ...) -> None: ...
    class Log(_message.Message):
        __slots__ = ("node_id", "level", "msg")
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        LEVEL_FIELD_NUMBER: _ClassVar[int]
        MSG_FIELD_NUMBER: _ClassVar[int]
        node_id: str
        level: str
        msg: str
        def __init__(self, node_id: _Optional[str] = ..., level: _Optional[str] = ..., msg: _Optional[str] = ...) -> None: ...
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RESEARCHER_ID_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    researcher_id: str
    scalar: FeedbackMessage.Scalar
    log: FeedbackMessage.Log
    def __init__(self, protocol_version: _Optional[str] = ..., researcher_id: _Optional[str] = ..., scalar: _Optional[_Union[FeedbackMessage.Scalar, _Mapping]] = ..., log: _Optional[_Union[FeedbackMessage.Log, _Mapping]] = ...) -> None: ...

class TaskRequest(_message.Message):
    __slots__ = ("node", "protocol_version")
    NODE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    node: str
    protocol_version: str
    def __init__(self, node: _Optional[str] = ..., protocol_version: _Optional[str] = ...) -> None: ...

class TaskResponse(_message.Message):
    __slots__ = ("size", "iteration", "bytes_")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    size: int
    iteration: int
    bytes_: bytes
    def __init__(self, size: _Optional[int] = ..., iteration: _Optional[int] = ..., bytes_: _Optional[bytes] = ...) -> None: ...

class TaskResult(_message.Message):
    __slots__ = ("size", "iteration", "bytes_")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    size: int
    iteration: int
    bytes_: bytes
    def __init__(self, size: _Optional[int] = ..., iteration: _Optional[int] = ..., bytes_: _Optional[bytes] = ...) -> None: ...
