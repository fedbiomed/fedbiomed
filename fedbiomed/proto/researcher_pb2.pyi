from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class FeedbackMessage(_message.Message):
    __slots__ = ["scalar", "log"]
    class Scalar(_message.Message):
        __slots__ = ["researcher_id", "node_id", "job_id", "train", "test", "test_on_local_updates", "test_on_global_updates", "metric", "epoch", "total_samples", "batch_samples", "num_batches", "num_samples_trained", "iteration", "protocol_version"]
        class MetricEntry(_message.Message):
            __slots__ = ["key", "value"]
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: float
            def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
        RESEARCHER_ID_FIELD_NUMBER: _ClassVar[int]
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        JOB_ID_FIELD_NUMBER: _ClassVar[int]
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
        PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
        researcher_id: str
        node_id: str
        job_id: str
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
        protocol_version: str
        def __init__(self, researcher_id: _Optional[str] = ..., node_id: _Optional[str] = ..., job_id: _Optional[str] = ..., train: bool = ..., test: bool = ..., test_on_local_updates: bool = ..., test_on_global_updates: bool = ..., metric: _Optional[_Mapping[str, float]] = ..., epoch: _Optional[int] = ..., total_samples: _Optional[int] = ..., batch_samples: _Optional[int] = ..., num_batches: _Optional[int] = ..., num_samples_trained: _Optional[int] = ..., iteration: _Optional[int] = ..., protocol_version: _Optional[str] = ...) -> None: ...
    class Log(_message.Message):
        __slots__ = ["researcher_id", "log"]
        RESEARCHER_ID_FIELD_NUMBER: _ClassVar[int]
        LOG_FIELD_NUMBER: _ClassVar[int]
        researcher_id: str
        log: str
        def __init__(self, researcher_id: _Optional[str] = ..., log: _Optional[str] = ...) -> None: ...
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    scalar: FeedbackMessage.Scalar
    log: FeedbackMessage.Log
    def __init__(self, scalar: _Optional[_Union[FeedbackMessage.Scalar, _Mapping]] = ..., log: _Optional[_Union[FeedbackMessage.Log, _Mapping]] = ...) -> None: ...

class Log(_message.Message):
    __slots__ = ["log"]
    LOG_FIELD_NUMBER: _ClassVar[int]
    log: str
    def __init__(self, log: _Optional[str] = ...) -> None: ...

class TaskRequest(_message.Message):
    __slots__ = ["node"]
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: str
    def __init__(self, node: _Optional[str] = ...) -> None: ...

class TaskResponse(_message.Message):
    __slots__ = ["size", "iteration", "bytes_"]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    size: int
    iteration: int
    bytes_: bytes
    def __init__(self, size: _Optional[int] = ..., iteration: _Optional[int] = ..., bytes_: _Optional[bytes] = ...) -> None: ...

class TaskReplyMessage(_message.Message):
    __slots__ = ["size", "iteration", "bytes_"]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    size: int
    iteration: int
    bytes_: bytes
    def __init__(self, size: _Optional[int] = ..., iteration: _Optional[int] = ..., bytes_: _Optional[bytes] = ...) -> None: ...

class TaskResponseUnary(_message.Message):
    __slots__ = ["bytes_"]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    bytes_: bytes
    def __init__(self, bytes_: _Optional[bytes] = ...) -> None: ...
