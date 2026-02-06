# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Global logger for fedbiomed

Written above origin Logger class provided by python.

Following features were added from to the original module:

- provides a logger instance of FedLogger, which is also a singleton, so it can be used "as is"
- provides a dedicated file handler
- provides a JSON/gRPC handler
(this permit to send error messages from a node to a researcher)
- works on python scripts / ipython / notebook
- manages a dictionary of handlers. Default keys are 'CONSOLE', 'GRPC', 'FILE',
  but any key is allowed (only one handler by key)
- allow changing log level globally, or on a specific handler (using its key)
- log levels can be provided as string instead of logging.* levels (no need to
  import logging in caller's code) just as in the initial python logger

**A typical usage is:**

```python
from fedbiomed.common.logger import logger

logger.info("information message")
```

All methods of the original python logger are provided. To name a few:

- logger.debug()
- logger.info()
- logger.warning()
- logger.error()
- logger.critical()

Contrary to other Fed-BioMed classes, the API of FedLogger is compliant with the coding conventions used for logger
(lowerCameCase)

!!! info "Dependency issue"
    Please pay attention to not create dependency loop then importing other fedbiomed package
"""

import inspect
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Callable, Optional

from fedbiomed.common.ipython import is_ipython
from fedbiomed.common.singleton import SingletonMeta

# default values
DEFAULT_LOG_FILE = "mylog.log"
DEFAULT_SECURITY_LOG_FILE = "security_audit.log"
DEFAULT_LOG_LEVEL = logging.WARNING
LOG_PREFIX = "%(prefix)s"
DEFAULT_FORMAT = f"%(asctime)s %(name)s{LOG_PREFIX} %(levelname)s - %(message)s"

# --- Security context (propagates across modules) ---
SECURITY_CONTEXT: ContextVar[dict] = ContextVar(
    "fedbiomed_security_context", default=None
)


# --- Helper functions for security logging ---
def _utc_timestamp() -> str:
    """UTC timestamp in ISO8601 with Z suffix (no milliseconds)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# --- No additional sanitization needed; json.dumps() handles serialization ---


class _SecurityOnlyFilter(logging.Filter):
    """Ensures only security events go to the SECURITY_FILE handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "is_security", False))


class _SecurityFormatter(logging.Formatter):
    """Formats security log records as JSON with consistent structure."""

    def __init__(self, security_defaults: dict):
        super().__init__()
        self._security_defaults = security_defaults

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        msg = record.getMessage()

        # If message is already JSON (from security_event), return as-is
        try:
            json.loads(msg)
            return msg
        except (json.JSONDecodeError, ValueError):
            pass

        # Otherwise, build JSON structure from log record
        ctx = dict(SECURITY_CONTEXT.get() or {})

        entry = {
            "timestamp": _utc_timestamp(),
            "node_id": self._security_defaults.get("node_id"),
            "node_name": self._security_defaults.get("node_name"),
            "researcher_id": ctx.get("researcher_id")
            or getattr(record, "researcher_id", None),
            "operation": ctx.get("operation") or getattr(record, "operation", None),
            "status": getattr(record, "status", "logged"),
            "fedbiomed_version": self._security_defaults.get("fedbiomed_version"),
            "caller_function": record.funcName,
            "caller_module": os.path.basename(record.pathname),
            "caller_file": record.pathname,
            "caller_line": record.lineno,
            "level": record.levelname,
            "message": msg,
        }

        # Merge context fields
        for key, value in ctx.items():
            if key not in entry or entry[key] is None:
                entry[key] = value

        return json.dumps(entry, default=str, separators=(",", ":"))


class _GrpcFormatter(logging.Formatter):
    """gRPC log  formatter"""

    # ATTENTION: should not be imported from this module

    def __init__(self, node_id):
        super().__init__()
        self._node_id = node_id

    def format(self, record):
        """Formats the message/data that is going to be send to remote party through gRPC"""

        json_message = {
            "asctime": record.__dict__["asctime"],
            "node_id": self._node_id,
            "name": record.__dict__["name"],
            "level": record.__dict__["levelname"],
            "message": record.__dict__["message"],
        }

        record.msg = json.dumps(json_message)
        return super().format(record)


class _GrpcHandler(logging.Handler):
    """Logger handler for GRPC connections

    This class handles the log messages that are tagged as to be sent to
    researcher.
    """

    def __init__(
        self,
        on_log: Callable,
        node_id: str = None,
    ) -> None:
        """Constructor

        Args:
            on_log: Method to call to send log to researcher
            node_id: unique node id
        """

        logging.Handler.__init__(self)
        self._node_id = node_id
        self._on_log = on_log

    def emit(self, record: Any):
        """Emits the logged record

        Args:
            record: is automatically passed by the logger class
        """

        if hasattr(record, "broadcast") or hasattr(record, "researcher_id"):
            msg = dict(
                level=record.__dict__["levelname"],
                msg=self.format(record),
                node_id=self._node_id,
            )

            # import is done here to avoid circular import it must also be done each time emit() is called
            import fedbiomed.common.message as message

            feedback = message.FeedbackMessage(
                researcher_id=record.researcher_id, log=message.Log(**msg)
            )

            try:
                self._on_log(feedback, record.broadcast)
            except Exception:
                logging.error("Not able to send log message to remote party")


class _IpythonConsoleHandler(logging.Handler):
    """Logger handler for Ipython consoles.

    Do not use in other contexts.
    """

    def emit(self, record: logging.LogRecord):
        """Emits the logged record

        Args:
            record: message emitted by the logger
        """
        from IPython.display import display

        display({"text/plain": self.format(record)}, raw=True)


class FedLogger(metaclass=SingletonMeta):
    """Base class for the logger.

    It uses python logging module by composition (only log() method is overwritten)

    All methods from the logging module can be accessed through the _logger member of the class if necessary
    (instead of overloading all the methods) (ex:  logger._logger.getEffectiveLevel() )

    Should not be imported
    """

    def __init__(self, level: str = DEFAULT_LOG_LEVEL):
        """Constructor of base class

        An initial console logger is installed (so the logger has at minimum one handler)

        Args:
            level: initial loglevel. This loglevel will be the default for all handlers, if called
                without the default level


        """

        # internal tables
        # transform string to logging.level
        self._nameToLevel = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # transform logging.level to string
        self._levelToName = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }

        # name this logger
        self._logger = logging.getLogger("fedbiomed")

        # Do not propagate (avoids log duplication when third party libraries uses logging module)
        self._logger.propagate = False

        self._default_level = DEFAULT_LOG_LEVEL  # MANDATORY ! KEEP THIS PLEASE !!!
        self._default_level = self._internal_level_translator(level)

        self._logger.setLevel(self._default_level)

        # Store base format used by handlers
        self._original_format = {}

        # init the handlers list and add a console handler on startup
        self._handlers = {}
        self.add_console_handler()

        # --- Security log defaults (filled by Node at startup) ---
        self._security_defaults = {
            "node_id": None,
            "fedbiomed_version": None,
        }

        pass

    def configure_security(
        self,
        *,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        fedbiomed_version: Optional[str] = None,
    ) -> None:
        """Configure default fields that must appear in every security log entry."""
        if node_id is not None:
            self._security_defaults["node_id"] = node_id
        if node_name is not None:
            self._security_defaults["node_name"] = node_name
        if fedbiomed_version is not None:
            self._security_defaults["fedbiomed_version"] = fedbiomed_version

    @contextmanager
    def security_context(self, **ctx: Any):
        """
        Bind security context for downstream logging.

        How reset works:
        - SECURITY_CONTEXT.set(new_dict) returns a token referencing the previous value.
        - In the `finally` block we call SECURITY_CONTEXT.reset(token) so the old context
          is restored even if exceptions occur (prevents context leaks).
        """
        current = dict(
            SECURITY_CONTEXT.get() or {}
        )  # copy to avoid mutation across calls
        current.update({k: v for k, v in ctx.items() if v is not None})
        token = SECURITY_CONTEXT.set(current)
        try:
            yield
        finally:
            SECURITY_CONTEXT.reset(token)

    def add_security_file_handler(
        self,
        filename: str = DEFAULT_SECURITY_LOG_FILE,
        level: Any = logging.INFO,
    ) -> None:
        """
        Adds a dedicated SECURITY_FILE handler that writes JSONL only.
        Only records emitted with extra {"is_security": True} will be written.
        Automatically rotates daily at midnight. Old logs are never deleted.

        Args:
            filename: Security log file path. Defaults to 'security_audit.log'
            level: Logging level for security events. Defaults to INFO
        """
        handler = TimedRotatingFileHandler(
            filename=filename,
            when="midnight",
            interval=1,
            backupCount=0,  # Keep all old logs, never delete
        )
        handler.setLevel(self._internal_level_translator(level))
        handler.setFormatter(_SecurityFormatter(self._security_defaults))
        handler.addFilter(_SecurityOnlyFilter())
        # Disable buffering to ensure immediate writes
        handler.stream.reconfigure(line_buffering=True)

        # Register under its own key so it doesn't collide with FILE handler
        self._internal_add_handler("SECURITY_FILE", handler, "%(message)s")

    def security_event(
        self,
        *,
        operation: Optional[str] = None,
        status: Optional[str] = None,
        researcher_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """
        Writes one JSON security/audit log line to security file only.

        Always present:
          - node_id, researcher_id, timestamp, operation, status, fedbiomed_version
          - caller_function, caller_module, caller_file, caller_line (automatically captured)

        Values resolved as:
          - explicit args > bound SECURITY_CONTEXT > defaults (for node_id/version)

        Args:
            operation: Name of the operation being logged (e.g., 'dataset_search', 'training_execute')
            status: Status of the operation (e.g., 'success', 'failure', 'pending')
            researcher_id: ID of the researcher performing the operation
            **fields: Additional fields to log (e.g., dataset_id, experiment_id, node_id)
        """
        ctx = dict(SECURITY_CONTEXT.get() or {})

        op = operation or ctx.get("operation")
        st = status or ctx.get("status")
        rid = researcher_id or ctx.get("researcher_id")

        # Capture caller information from the stack
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        caller_info = {
            "caller_function": caller_frame.f_code.co_name
            if caller_frame
            else "unknown",
            "caller_module": os.path.basename(caller_frame.f_code.co_filename)
            if caller_frame
            else "unknown",
            "caller_file": caller_frame.f_code.co_filename
            if caller_frame
            else "unknown",
            "caller_line": caller_frame.f_lineno if caller_frame else 0,
        }

        entry = {
            "timestamp": _utc_timestamp(),
            "node_id": self._security_defaults.get("node_id"),
            "node_name": self._security_defaults.get("node_name"),
            "researcher_id": rid,
            "operation": op,
            "status": st,
            "fedbiomed_version": self._security_defaults.get("fedbiomed_version"),
            **caller_info,
        }

        # Merge extra fields: context first, then explicit fields override
        merged = {}
        merged.update(ctx)
        merged.update(fields)

        # Avoid duplicating required keys in payload
        for k in ("operation", "status", "researcher_id"):
            merged.pop(k, None)

        entry.update(merged)

        # Write only to security file (is_security=True filters out other handlers)
        self._logger.info(
            json.dumps(entry, default=str, separators=(",", ":")),
            extra={"is_security": True},
        )

    def _internal_add_handler(
        self, output: str, handler: Callable, format: Optional[str] = None
    ):
        """Private method

        Add a handler to the logger. only one handler is allowed
        for a given output (type)

        Args:
            output: Tag for the logger ("CONSOLE", "FILE"), this is a string used as a hash key
            handler: Proper handler to install. if handler is None, it will remove the previous installed handler
            format: format string for this handler
        """
        if handler is None:
            if output in self._handlers:
                self.removeHandler(self._handlers[output])
                del self._handlers[output]
                del self._original_format[output]
                self._logger.debug(" removing handler for: " + output)
            return

        if output not in self._handlers:
            self._logger.debug(" adding handler for: " + output)
            self._handlers[output] = handler
            self._logger.addHandler(handler)
            self._handlers[output].setLevel(self._default_level)
            self._original_format[output] = format
        else:
            self._logger.warning(output + " handler already present - ignoring")

        pass

    def _internal_level_translator(self, level: Any = DEFAULT_LOG_LEVEL) -> Any:
        """Private method

        This helper allows to use a string instead of logging.* then using logger levels. This is the opposite
        of the getLevelName() of logging python module and this is not provided by the logging.Logger class

        Args:
            level:  wanted parameter

        Returns:
            logging.* form of the level
        """

        # logging.*
        if level in self._levelToName:
            return level

        # strings
        if isinstance(level, str):
            upper_level = level.upper()
            if upper_level in self._nameToLevel:
                return self._nameToLevel[upper_level]

        self._logger.warning("Calling setLevel() with bad value: " + str(level))
        self._logger.warning(
            "Setting " + self._levelToName[DEFAULT_LOG_LEVEL] + " level instead"
        )

        return DEFAULT_LOG_LEVEL

    def add_file_handler(
        self,
        filename: str = DEFAULT_LOG_FILE,
        format: str = DEFAULT_FORMAT,
        level: any = DEFAULT_LOG_LEVEL,
    ):
        """Adds a file handler

        Args:
            filename: File to log to
            format: Log format
            level: Initial level of the logger
        """

        handler = logging.FileHandler(filename=filename, mode="a")
        handler.setLevel(self._internal_level_translator(level))

        formatter = logging.Formatter(format.replace(LOG_PREFIX, ""))
        handler.setFormatter(formatter)

        self._internal_add_handler("FILE", handler, format)

    def add_console_handler(
        self, format: str = DEFAULT_FORMAT, level: Any = DEFAULT_LOG_LEVEL
    ):
        """Adds a console handler

        Args:
            format: the format string of the logger
            level: initial level of the logger for this handler (optional) if not given, the default level is set
        """
        if is_ipython():
            handler = _IpythonConsoleHandler()
        else:
            handler = logging.StreamHandler()

        handler.setLevel(self._internal_level_translator(level))

        formatter = logging.Formatter(format.replace(LOG_PREFIX, ""))
        handler.setFormatter(formatter)
        self._internal_add_handler("CONSOLE", handler, format)

        pass

    def add_grpc_handler(
        self, on_log: Callable = None, node_id: str = None, level: Any = logging.INFO
    ):
        """Adds a gRPC handler, to publish error message on a topic

        Args:
            on_log: Provided by higher level GRPC implementation
            node_id: id of the caller (necessary for msg formatting to the researcher)
            level: level of this handler (non-mandatory) level must be lower than ERROR to ensure that the
                research get all ERROR/CRITICAL messages
        """

        handler = _GrpcHandler(
            on_log=on_log,
            node_id=node_id,
        )

        # may be not necessary ?
        handler.setLevel(self._internal_level_translator(level))
        formatter = _GrpcFormatter(node_id)

        handler.setFormatter(formatter)
        self._internal_add_handler("GRPC", handler)

        # as a side effect this will set the minimal level to ERROR
        self.setLevel(level, "GRPC")

        pass

    def log(self, level: Any, msg: str):
        """Overrides the logging.log() method to allow the use of string instead of a logging.* level"""

        level = logger._internal_level_translator(level)
        self._logger.log(level, msg)

    def setLevel(self, level: Any, htype: Any = None):
        """Overrides the setLevel method, to deal with level given as a string and to change le level of
        one or all known handlers

        This also change the default level for all future handlers.

        !!! info "Remark"

            Level should not be lower than CRITICAL (meaning CRITICAL errors are always displayed)

            Example:
            ```python
            setLevel( logging.DEBUG, 'FILE')
            ```

        Args:
            level : level to modify, can be a string or a logging.* level (mandatory)
            htype : if provided (non-mandatory), change the level of the given handler. if not  provided (or None),
                change the level of all known handlers
        """

        level = self._internal_level_translator(level)

        if htype is None:
            # store this level (for future handler adding)
            self._logger.setLevel(level)

            for h in self._handlers:
                self._handlers[h].setLevel(level)
            return

        if htype in self._handlers:
            self._handlers[htype].setLevel(level)
            return

        # htype provided but no handler for this type exists
        self._logger.warning(htype + " handler not initialized yet")

    def setPrefix(self, prefix: str = "") -> None:
        """Sets a log prefix for all handlers.

        Args:
            prefix: Prefix to add to all log messages
        """
        for h in self._handlers:
            if self._original_format[h] is not None:
                self._handlers[h].setFormatter(
                    logging.Formatter(
                        self._original_format[h].replace(LOG_PREFIX, prefix),
                    )
                )

    def info(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Extends arguments of info message.

        Valid only GrpcHandler is existing

        Args:
            msg: Message to log
            broadcast: Broadcast message to all available researchers
            researcher_id: ID of the researcher that the message will be sent.
                If broadcast True researcher id will be ignored
            **kwargs: Additional keyword arguments. Can include extra={"is_security": True}
                to also write to security log file
        """
        # Merge extra dicts properly to support is_security flag
        extra = kwargs.pop("extra", {})
        extra.update({"researcher_id": researcher_id, "broadcast": broadcast})
        self._logger.info(
            msg,
            *args,
            **kwargs,
            extra=extra,
        )

    def debug(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        # Merge extra dicts properly to support is_security flag
        extra = kwargs.pop("extra", {})
        extra.update({"researcher_id": researcher_id, "broadcast": broadcast})
        self._logger.debug(
            msg,
            *args,
            **kwargs,
            extra=extra,
        )

    def warning(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        # Merge extra dicts properly to support is_security flag
        extra = kwargs.pop("extra", {})
        extra.update({"researcher_id": researcher_id, "broadcast": broadcast})
        self._logger.warning(
            msg,
            *args,
            **kwargs,
            extra=extra,
        )

    def critical(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        # Merge extra dicts properly to support is_security flag
        extra = kwargs.pop("extra", {})
        extra.update({"researcher_id": researcher_id, "broadcast": broadcast})
        self._logger.critical(
            msg,
            *args,
            **kwargs,
            extra=extra,
        )

    def error(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        # Merge extra dicts properly to support is_security flag
        extra = kwargs.pop("extra", {})
        extra.update({"researcher_id": researcher_id, "broadcast": broadcast})
        self._logger.error(
            msg,
            *args,
            **kwargs,
            extra=extra,
        )

    def __getattr__(self, s: Any):
        """Calls the method from self._logger if not override by this class"""

        try:
            return self.__getattribute__(s)
        except AttributeError:
            return self._logger.__getattribute__(s)


"""Instantiation of the logger singleton"""
logger = FedLogger()
