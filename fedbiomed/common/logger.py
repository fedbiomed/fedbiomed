# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Global logger for fedbiomed

Written above origin Logger class provided by python.

Following features were added from to the original module:

- provides a logger instance of FedLogger, which is also a singleton, so it can be used "as is"
- provides a dedicated file handler
- provides a JSON/MQTT handler: all messages with priority greater than error are sent to the MQQT handler
(this permit to send error messages from a node to a researcher)
- works on python scripts / ipython / notebook
- manages a dictionary of handlers. Default keys are 'CONSOLE', 'MQTT', 'FILE',
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

import json  # we do not use fedbiomed.common.json to avoid dependancy loops

import logging
import logging.handlers

from typing import Callable, Any
# these fedbiomed.* import are OK, they do not introduce dependancy loops
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedLoggerError
from fedbiomed.common.singleton import SingletonMeta

# default values
DEFAULT_LOG_FILE = 'mylog.log'
DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_LOG_TOPIC = 'general/logger'


class _MqttFormatter(logging.Formatter):
    """Mqtt  formatter """

    # ATTENTION: should not be imported from this module

    def __init__(self, node_id):
        super().__init__()
        self._node_id = node_id

    # fields of record
    #
    # name: 'fedbiomed'
    # msg: 'mqtt+console ERROR message'
    # args: ()
    # levelname: 'ERROR'
    # levelno: 40
    # pathname: '/.../common/logger.py'
    # filename: 'logger.py'
    # module: 'logger'
    # exc_info: None
    # exc_text: None
    # stack_info: None
    # lineno: 349
    # funcName: 'error'
    # created: 1631108190.796861
    # msecs: 796.860933303833
    # relativeCreated: 3110.7118129730225
    # thread: 4484275712
    # threadName: 'MainThread'
    # processName: 'MainProcess'
    # process: 41544
    # message: 'mqtt+console ERROR message'
    # asctime: '2021-09-08 15:36:30796'

    def format(self, record):
        json_message = {"asctime": record.__dict__["asctime"], "node_id": self._node_id,
                        "name": record.__dict__["name"], "level": record.__dict__["levelname"],
                        "message": record.__dict__["message"]}

        record.msg = json.dumps(json_message)
        return super().format(record)


#
# mqtt handler
#
class _MqttHandler(logging.Handler):
    """
    (internal) handler class to deal with MQTT

    should be imported
    """

    def __init__(self,
                 mqtt: Any = None,
                 node_id: str = None,
                 topic: str = DEFAULT_LOG_TOPIC
                 ):
        """
        Constructor

        Args:
            mqtt: opened MQTT object
            node_id: unique MQTT client id
            topic: topic/channel to publish to (default to logging.WARNING)
        """

        logging.Handler.__init__(self)
        self._node_id = node_id
        self._mqtt = mqtt
        self._topic = topic

    def emit(self, record: Any):
        """Do the proper job (override the logging.Handler method() )

        Args:
            record: is automatically passed by the logger class
        """

        # format a message as expected for LogMessage
        # TODO:
        # - get the researcher_id from the caller (is it needed ???)
        #   researcher_id is not known then adding the mqtt handler....

        msg = dict(command='log',
                   level=record.__dict__["levelname"],
                   msg=self.format(record),
                   node_id=self._node_id,
                   researcher_id='<unknown>')
        try:
            # import is done here to avoid circular import it must also be done each time emit() is called
            import fedbiomed.common.message as message

            # verify the message content with Message validator
            _ = message.NodeMessages.reply_create(msg)
            self._mqtt.publish(self._topic, json.dumps(msg))
        except Exception:  # pragma: no cover
            # obviously cannot call logger here... (infinite loop)  cannot also send the message to the researcher
            # (which was the purpose of the try block which failed)
            print(record.__dict__["asctime"],
                  record.__dict__["name"],
                  "CRITICAL - badly formatted MQTT log message. Cannot send MQTT message")
            _msg = ErrorNumbers.FB602.value + ": badly formatted MQTT log message. Cannot send MQTT message"
            raise FedbiomedLoggerError(_msg)


class FedLogger(metaclass=SingletonMeta):
    """Base class for the logger. it uses python logging module by composition (only log() method is overwritten)

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
        self._default_level = self._internalLevelTranslator(level)

        self._logger.setLevel(self._default_level)

        # init the handlers list and add a console handler on startup
        self._handlers = {}
        self.addConsoleHandler()

        pass

    def _internalAddHandler(self, output: str, handler: Callable):
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

    def _internalLevelTranslator(self, level: Any = DEFAULT_LOG_LEVEL) -> Any:
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
            upperlevel = level.upper()

            if upperlevel in self._nameToLevel:
                return self._nameToLevel[upperlevel]

        # bad input !

        # should always work, even at startup with _logger badly initialized
        # because this method is called by __init__
        # (where else to log this really ?)
        self._logger.warning("calling selLevel() with bad value: " + str(level))
        self._logger.warning("setting " + self._levelToName[DEFAULT_LOG_LEVEL] + " level instead")
        return DEFAULT_LOG_LEVEL

    def addFileHandler(self,
                       filename: str = DEFAULT_LOG_FILE,
                       format: str = '%(asctime)s %(name)s %(levelname)s - %(message)s',
                       level: any = DEFAULT_LOG_LEVEL):
        """Adds a file handler

        Args:
            filename: File to log to
            format: Log format
            level: Initial level of the logger (optionnal)
        """

        handler = logging.FileHandler(filename=filename, mode='a')
        handler.setLevel(self._internalLevelTranslator(level))

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        self._internalAddHandler("FILE", handler)
        pass

    def addConsoleHandler(self,
                          format: str = '%(asctime)s %(name)s %(levelname)s - %(message)s',
                          level: Any = DEFAULT_LOG_LEVEL):

        """Adds a console handler

        Args:
            format: the format string of the logger
            level: initial level of the logger for this handler (optional) if not given, the default level is set
        """

        handler = logging.StreamHandler()

        handler.setLevel(self._internalLevelTranslator(level))

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        self._internalAddHandler("CONSOLE", handler)

        pass

    def addMqttHandler(self,
                       mqtt: Any = None,
                       node_id: str = None,
                       topic: Any = DEFAULT_LOG_TOPIC,
                       level: Any = logging.ERROR
                       ):

        """Adds a mqtt handler, to publish error message on a topic

        Args:
            mqtt: already opened MQTT object
            node_id: id of the caller (necessary for msg formatting to the researcher)
            topic: topic to publish to (non-mandatory)
            level: level of this handler (non-mandatory) level must be lower than ERROR to ensure that the
                research get all ERROR/CRITICAL messages
        """

        handler = _MqttHandler(
            mqtt=mqtt,
            node_id=node_id,
            topic=topic
        )

        # may be not necessary ?
        handler.setLevel(self._internalLevelTranslator(level))
        formatter = _MqttFormatter(node_id)

        handler.setFormatter(formatter)
        self._internalAddHandler("MQTT", handler)

        # as a side effect this will set the minimal level to ERROR
        self.setLevel(level, "MQTT")
        pass

    def delMqttHandler(self):
        self._internalAddHandler("MQTT", None)

    def log(self, level: Any, msg: str):
        """Overrides the logging.log() method to allow the use of string instead of a logging.* level """

        level = logger._internalLevelTranslator(level)
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

        level = self._internalLevelTranslator(level)

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

    def __getattr__(self, s: Any):
        """Calls the method from self._logger if not override by this class"""

        try:
            _x = self.__getattribute__(s)
        except AttributeError:
            _x = self._logger.__getattribute__(s)
            return _x
        else:
            return _x  # pragma: no cover


"""Instantiation of the logger singleton"""
logger = FedLogger()
