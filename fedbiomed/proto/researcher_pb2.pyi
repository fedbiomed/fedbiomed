from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class FeedbackMessage(_message.Message):
    __slots__ = ["scalar", "log"]
    class AddScalar(_message.Message):
        __slots__ = ["bytes_"]
        BYTES__FIELD_NUMBER: _ClassVar[int]
        bytes_: bytes
        def __init__(self, bytes_: _Optional[bytes] = ...) -> None: ...
    class Log(_message.Message):
        __slots__ = ["log"]
        LOG_FIELD_NUMBER: _ClassVar[int]
        log: str
        def __init__(self, log: _Optional[str] = ...) -> None: ...
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    scalar: FeedbackMessage.AddScalar
    log: FeedbackMessage.Log
    def __init__(self, scalar: _Optional[_Union[FeedbackMessage.AddScalar, _Mapping]] = ..., log: _Optional[_Union[FeedbackMessage.Log, _Mapping]] = ...) -> None: ...

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

class TaskResponseUnary(_message.Message):
    __slots__ = ["bytes_"]
    BYTES__FIELD_NUMBER: _ClassVar[int]
    bytes_: bytes
    def __init__(self, bytes_: _Optional[bytes] = ...) -> None: ...
