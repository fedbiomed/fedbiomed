"""
Global logger for fedbiomed

Written above origin Logger class provided by python

Add the following features:
- provides a file handler
- provides a JSON MQTT handler
- works on python scripts / ipython / notebook
- manages handlers with a key name. Default keys are 'CONSOLE', 'MQTT', 'FILE',
  but any key is allowed (only oen handler by key)
- allow to change log level globally, or on a specific handler (using its key)
- log levels can be provided as string instead of logging.* levels (no need to
   import logging in caller's code)
"""

import paho.mqtt.publish as publish

import copy
import json
import sys
import time

import logging
import logging.handlers

from fedbiomed.common.singleton import SingletonMeta

# default values
DEFAULT_LOG_FILE   = 'mylog.log'
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_TOPIC  = 'general/logger'

#
# mqtt  formatter
#
class MqttFormatter(logging.Formatter):

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

        json_message = {
            "asctime"   : record.__dict__["asctime"],
            "node_id"   : self._node_id
        }
        json_message["name"] = record.__dict__["name"]
        json_message["level"] = record.__dict__["levelname"]
        json_message["message"] = record.__dict__["message"]

        record.msg = json.dumps(json_message)
        return super().format(record)


#
# mqtt handler
#
class MqttHandler(logging.Handler):
    """
    A handler class to deal with MQTT
    """

    def __init__(self,
                 mqtt        = None,
                 node_id     = None,
                 topic       = DEFAULT_LOG_TOPIC
                 ):
        """
        Constructor

        parameters:
        mqtt      : opened MQTT object
        node_id   : unique MQTT client id
        topic     : topic/channel to publish to (default to logging.WARNING)
        """

        logging.Handler.__init__(self)
        self._node_id        = node_id
        self._mqtt           = mqtt
        self._topic          = topic

    def emit(self, record):
        """
        do the proper job (override the logging.Handler method() )

        parameters:
        record : is automatically passed by the logger class
        """

        #
        # format a message as expected for LogMessage
        #
        # TODO:
        # - get the researcher_id from the caller (is it needed ???)
        #   researcher_id is not known then adding the mqtt handler....
        #
        msg = dict(
            command       = 'log',
            level         = record.__dict__["levelname"],
            msg           = self.format(record),
            node_id       = self._node_id,
            researcher_id = '<unknown>'
        )
        try:
            #
            # import is done here to avoid circular import
            # it must also be done each time emit() is called
            #
            import fedbiomed.common.message as message

            # verify the message content with Message validator
            r = message.NodeMessages.reply_create( msg )
            self._mqtt.publish(self._topic, json.dumps(msg))

        except:
            # obviously cannot call logger here... (infinite loop)
            print(
                record.__dict__["asctime"],
                record.__dict__["name"],
                "CRITICAL - Badly formatted MQTT log message. Cannot send MQTT message"
            )
            sys.exit(-1)


class _LoggerBase():
    """
    base class for the logger. it uses python logging module by
    composition

    debug/../critical methods are overrided

    all methods from the logging module can be accessed through
    the _logger member of the class if necessary (instead of overloading all the methods)
    (ex:  logger._logger.getEffectiveLevel() )
    """


    def __init__(self, level = DEFAULT_LOG_LEVEL ):
        """
        constructor of base class

        parameter: initial loglevel.
        This loglevel will be the default for all handlers, if called
        without the default level

        an initial console logger is installed (so the logger has at minimum one handler)
        """

        # internal tables
        # transform string to logging.level
        self._nameToLevel = {
            "DEBUG"          : logging.DEBUG,
            "INFO"           : logging.INFO,
            "WARNING"        : logging.WARNING,
            "ERROR"          : logging.ERROR,
            "CRITICAL"       : logging.CRITICAL,
        }

        # transform logging.level to string
        self._levelToName = {
            logging.DEBUG    : "DEBUG",
            logging.INFO     : "INFO",
            logging.WARNING  : "WARNING",
            logging.ERROR    : "ERROR",
            logging.CRITICAL : "CRITICAL"
        }

        # name this logger
        self._logger = logging.getLogger("fedbiomed")

        self._default_level = DEFAULT_LOG_LEVEL  # MANDATORY ! KEEP THIS PLEASE !!!
        self._default_level = self._internalLevelTranslator(level)

        self._logger.setLevel(self._default_level)

        # init the handlers list and add a console handler on startup
        self._handlers = {}
        self.addConsoleHandler()

        pass


    def _internalAddHandler(self, output , handler):
        """
        private method

        add a handler to the logger. only one handler is allowed
        for a given output (type)

        parameters:
        output  = tag for the logger ("CONSOLE", "FILE"), this is a string used as an hash key
        handler = proper handler to install
        """
        if output not in self._handlers:
            self._logger.debug(" adding handler: " + output)
            self._handlers[output] = handler
            self._logger.addHandler(handler)
            self._handlers[output].setLevel( self._default_level)
        else:
            self._logger.warning(output + " handler already present - ignoring")

        pass


    def _internalLevelTranslator(self, level = DEFAULT_LOG_LEVEL) :
        """
        private method

        this helper allows to use a string instead of logging.* then using logger levels

        parameter:
        level:  wanted parameter

        output:
        level:  logging.* form of the level

        ex:
        _internalLevelTranslator('DEBUG')  returns logging.DEBUG

        remark:
        this is the opposite of the getLevelName() of logging python module
        and this is not provided by the logging.Logger class
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
        return DEFAULT_LOG_LEVEL


    def _internalLevelToString(self, level):
        """
        Returns a string corresponding to the log level
        """
        if level in self._string_levels:
            return level
        if level in self._original_levels:
            return self._original_levels[level]
        return "UNKNOWN"


    def addFileHandler(self,
                       filename = DEFAULT_LOG_FILE,
                       format = '%(asctime)s %(name)s %(levelname)s - %(message)s' ,
                       level = DEFAULT_LOG_LEVEL):
        """
        add a file handler

        parameters:
        filename : file to log to
        level    : initial level of the logger (optionnal)
        """

        handler  = logging.FileHandler(filename=filename, mode='a')
        handler.setLevel( self._internalLevelTranslator(level) )

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        self._internalAddHandler("FILE", handler)
        pass


    def addConsoleHandler(self,
                          format = '%(asctime)s %(name)s %(levelname)s - %(message)s' ,
                          level  = DEFAULT_LOG_LEVEL):

        """
        add a console handler

        parameters:
        format : the format string of the logger
        level  : initial level of the logger for this handler (optionnal)
                 if not given, the default level is set
        """

        handler = logging.StreamHandler()

        handler.setLevel( self._internalLevelTranslator(level) )

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        self._internalAddHandler("CONSOLE", handler)

        pass


    def addMqttHandler(self,
                       mqtt        = None,
                       node_id     = None,
                       topic       = DEFAULT_LOG_TOPIC,
                       level       = logging.ERROR
                       ):

        """
        add a mqtt handler, to publish error message on a topic

        parameters:
        mqtt        : already opened MQTT object
        node_id     : id of the caller (necessary for msg formatting to the researcher)
        topic       : topic to publish to    (non mandatory)
        level       : level of this handler  (non mandatory)
                      level must be lower than ERROR to insure that the
                      research get all ERROR/CRITICAL messages
        """

        handler = MqttHandler(
            mqtt        = mqtt,
            node_id     = node_id ,
            topic       = topic
        )

        # may be not necessary ?
        handler.setLevel( self._internalLevelTranslator(level) )
        formatter = MqttFormatter(node_id)

        handler.setFormatter(formatter)
        self._internalAddHandler("MQTT", handler)

        # as a side effect this will set the minimal level to ERROR
        self.setLevel(level , "MQTT")
        pass


    def log(self, level, msg):
        """
        overrides the logging.log() method to allow the use of
        string instead of a logging.* level
        """

        level = logger._internalLevelTranslator(level)
        self._logger.log(
            level,
            msg
        )


    def setLevel(self, level, htype = None ):
        """
        overrides the setLevel method, to deal with level given as a string
        and to change le level of one or all known handlers

        this also change the default level for all future handlers.

        parameter:
        level : level to modify, can be a string or a logging.* level (mandatory)
        htype : if provided (non madatory), change the level of the given handler.
                if not  provided (or None), change the level of all known handlers

        Remark: level should not be lower than CRITICAL (meaning CRITICAL errors are
                always displayed

        Ex:
            setLevel( logging.DEBUG, 'FILE')
        """

        level = self._internalLevelTranslator(level)

        if level > logging.CRITICAL:
            level = logging.critical
            logger.debug("setting minimal level to CRITICAL")

        # store this level (for future handler adding)
        self._logger.setLevel( level )

        if htype is None:
            for h in self._handlers:
                self._handlers[h].setLevel(level)
            return

        if htype in self._handlers:
            self._handlers[htype].setLevel(level)
            return

        # htype provided but no handler for this type exists
        self._logger.warning(htype + " handler not initialized yet")



    def __getattr__(self,s):
        """
        call the method from self._logger if not overrided by this class
        """
        try:
            _x = self.__getattribute__(s)
        except AttributeError:
            _x = self._logger.__getattribute__(s)
            return _x
        else:
            return _x # pragma: no cover


#
# this is the proper Logger to use
class _FedLogger(_LoggerBase, metaclass=SingletonMeta):
    pass


logger = _FedLogger()
