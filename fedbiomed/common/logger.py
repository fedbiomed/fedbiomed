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

import json
import logging
import logging.handlers

from typing import Callable, Any

from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.ipython import is_ipython

# default values
DEFAULT_LOG_FILE = 'mylog.log'
DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_FORMAT = '%(asctime)s %(name)s %(levelname)s - %(message)s'


class _GrpcFormatter(logging.Formatter):
    """gRPC log  formatter """

    # ATTENTION: should not be imported from this module

    def __init__(self, node_id):
        super().__init__()
        self._node_id = node_id

    def format(self, record):
        """Formats the message/data that is going to be send to remote party through gRPC"""

        json_message = {"asctime": record.__dict__["asctime"], "node_id": self._node_id,
                        "name": record.__dict__["name"], "level": record.__dict__["levelname"],
                        "message": record.__dict__["message"]}

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

        if hasattr(record, 'broadcast') or hasattr(record, 'researcher_id'):


            msg = dict(
                level=record.__dict__["levelname"],
                msg=self.format(record),
                node_id=self._node_id)

            # import is done here to avoid circular import it must also be done each time emit() is called
            import fedbiomed.common.message as message
            feedback = message.FeedbackMessage(researcher_id=record.researcher_id, log=message.Log(**msg))

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

        # `display` is defined in Ipython context
        display({'text/plain': self.format(record)}, raw=True)


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
            logging.CRITICAL: "CRITICAL"
        }

        # name this logger
        self._logger = logging.getLogger("fedbiomed")

        # Do not propagate (avoids log duplication when third party libraries uses logging module)
        self._logger.propagate = False

        self._default_level = DEFAULT_LOG_LEVEL  # MANDATORY ! KEEP THIS PLEASE !!!
        self._default_level = self._internal_level_translator(level)

        self._logger.setLevel(self._default_level)

        # init the handlers list and add a console handler on startup
        self._handlers = {}
        self.add_console_handler()

        pass

    def _internal_add_handler(self, output: str, handler: Callable):
        """Private method

        Add a handler to the logger. only one handler is allowed
        for a given output (type)

        Args:
            output: Tag for the logger ("CONSOLE", "FILE"), this is a string used as a hash key
            handler: Proper handler to install. if handler is None, it will remove the previous installed handler
        """
        if handler is None:
            if output in self._handlers:
                self.removeHandler(self._handlers[output])
                del self._handlers[output]
                self._logger.debug(" removing handler for: " + output)
            return

        if output not in self._handlers:
            self._logger.debug(" adding handler for: " + output)
            self._handlers[output] = handler
            self._logger.addHandler(handler)
            self._handlers[output].setLevel(self._default_level)
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

        self._logger.warning("Calling selLevel() with bad value: " + str(level))
        self._logger.warning("Setting " + self._levelToName[DEFAULT_LOG_LEVEL] + " level instead")

        return DEFAULT_LOG_LEVEL

    def add_file_handler(
            self,
            filename: str = DEFAULT_LOG_FILE,
            format: str = DEFAULT_FORMAT,
            level: any = DEFAULT_LOG_LEVEL):
        """Adds a file handler

        Args:
            filename: File to log to
            format: Log format
            level: Initial level of the logger
        """

        handler = logging.FileHandler(filename=filename, mode='a')
        handler.setLevel(self._internal_level_translator(level))

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        self._internal_add_handler("FILE", handler)


    def add_console_handler(self,
                            format: str = DEFAULT_FORMAT,
                            level: Any = DEFAULT_LOG_LEVEL):

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

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        self._internal_add_handler("CONSOLE", handler)

        pass

    def add_grpc_handler(self,
                         on_log: Callable = None,
                         node_id: str = None,
                         level: Any = logging.INFO
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
        """Overrides the logging.log() method to allow the use of string instead of a logging.* level """

        level = logger._internal_level_translator(level)
        self._logger.log(
            level,
            msg
        )

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


    def info(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Extends arguments of info message.

        Valid only GrpcHandler is existing

        Args:
            msg: Message to log
            broadcast: Broadcast message to all available researchers
            researcher_id: ID of the researcher that the message will be sent.
                If broadcast True researcher id will be ignored
        """
        self._logger.info(msg, *args, **kwargs,
                          extra={"researcher_id": researcher_id, 'broadcast': broadcast})

    def debug(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        self._logger.debug(msg, *args, **kwargs,
                           extra={"researcher_id": researcher_id, 'broadcast': broadcast})

    def warning(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        self._logger.warning(msg, *args, **kwargs,
                             extra={"researcher_id": researcher_id, 'broadcast': broadcast})

    def critical(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        self._logger.critical(msg, *args, **kwargs,
                              extra={"researcher_id": researcher_id, 'broadcast': broadcast})

    def error(self, msg, *args, broadcast=False, researcher_id=None, **kwargs):
        """Same as info message"""
        self._logger.error(msg, *args, **kwargs,
                           extra={"researcher_id": researcher_id, 'broadcast': broadcast})


    def __getattr__(self, s: Any):
        """Calls the method from self._logger if not override by this class"""

        try:
            return self.__getattribute__(s)
        except AttributeError:
            return self._logger.__getattribute__(s)


"""Instantiation of the logger singleton"""
logger = FedLogger()
