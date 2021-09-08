"""
Global logger for fedbiomed

Written above origin Logger class provided by python

Add the following features:
- provides a JSON file handler
- provides a JSON MQTT handler
- works on python scripts / ipython / notebook
- manages handlers with a key name. Default keys are 'CONSOLE', 'MQTT', 'FILE',
  but any key is allowed (only oen handler by key)
- allow to change log level globally, or on a specific handler (using its key)
- log levels can be provided as string instead of logging.* levels (no need to
   import logging in caller's code)
"""

import paho.mqtt.publish as publish

import logging
import logging.handlers

import json_log_formatter

# default values
DEFAULT_LOG_FILE   = 'mylog.log'
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_TOPIC  = 'general/logger'

#
# mqtt handler
#
class MqttHandler(logging.Handler):
    """
    A handler class to deal with MQTT
    """

    def __init__(self,
                 mqtt        = None,
                 client_id   = None,
                 topic       = DEFAULT_LOG_TOPIC
                 ):
        """
        Constructor

        parameters:
        mqtt      : opened MQTT object
        client_id : unique MQTT client id
        topic     : topic/channel to publish to (default to logging.WARNING)
        """

        logging.Handler.__init__(self)
        self._client_id = client_id
        self._mqtt      = mqtt
        self._topic     = topic


    def emit(self, record):
        """
        do the proper job (override the logging.Handler method() )

        parameters:
        record : is automatically passed by the logger class
        """

        msg = self.format(record)
        self._mqtt.publish(self._topic, msg)

#
# singletonizer: transforms a class to a singleton
# nothing else to say really !
class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


#
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
        self._levels = {
            "DEBUG"          : logging.DEBUG,
            "INFO"           : logging.INFO,
            "WARNING"        : logging.WARNING,
            "ERROR"          : logging.ERROR,
            "CRITICAL"       : logging.CRITICAL,
        }

        # transform logging.level to string
        self._original_levels = {
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
        """

        # logging.*
        if level in self._original_levels:
            return level

        # strings
        if isinstance(level, str):
            upperlevel = level.upper()

            if upperlevel in self._levels:
                return self._levels[upperlevel]

        # bad input !

        # should always work, even at startup with _logger badly initialized
        # because this method is called by __init__
        # (where else to log this really ?)
        self._logger.warning("calling selLevel() with bad value: " + str(level))
        return DEFAULT_LOG_LEVEL



    def addJsonFileHandler(self, filename = DEFAULT_LOG_FILE, level = DEFAULT_LOG_LEVEL):
        """
        add a JSON file handler

        parameters:
        filename : file to log to
        level    : initial level of the logger (optionnal)
        """

        handler  = logging.FileHandler(filename=filename, mode='a')
        handler.setLevel( self._internalLevelTranslator(level) )

        formatter = json_log_formatter.JSONFormatter()
        handler.setFormatter(formatter)

        self._internalAddHandler("FILE", handler)
        pass


    def addConsoleHandler(self,
                          format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' ,
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
                       client_id   = None,
                       topic       = DEFAULT_LOG_TOPIC,
                       level       = DEFAULT_LOG_LEVEL
                       ):

        """
        add a mqtt handler, to publish error message on a topic

        parameters:
        mqtt        : already opened MQTT object
        client_id   : unique client id of the caller
        topic       : topic to publish to    (non mandatory)
        level       : level of this handler  (non mandatory)
        """

        handler = MqttHandler(
            mqtt        = mqtt,
            client_id   = client_id ,
            topic       = topic
        )

        handler.setLevel( self._internalLevelTranslator(level) )
        formatter = json_log_formatter.JSONFormatter()

        handler.setFormatter(formatter)
        self._internalAddHandler("MQTT", handler)

        pass


    def log(self, level, msg):
        """
        overrides the logging.log() method to allow the usae of
        string instead of a logging.* level
        """

        level = logger._internalLevelTranslator(level)
        if isinstance(msg, dict):
            msg["level"] = level
            self._logger.log(level, "from_json_handler", extra = msg)
        else:
            self._logger.log(
                level,
                msg,
                extra = { "level": self._original_levels[level]}
            )


    def debug(self, msg):
        """
        overrides the logging.debug() method
        """
        if isinstance(msg, dict):
            msg["level"] = "DEBUG"
            self._logger.debug("from_json_handler", extra = msg)
        else:
            self._logger.debug(msg, extra = { "level": "DEBUG"} )


    def info(self, msg):
        """
        overrides the logging.info() method
        """
        if isinstance(msg, dict):
            msg["level"] = "INFO"
            self._logger.info("from_json_handler", extra = msg)
        else:
            self._logger.info(msg, extra = { "level": "INFO"} )


    def warning(self, msg):
        """
        overrides the logging.warning() method
        """
        if isinstance(msg, dict):
            msg["level"] = "WARNING"
            self._logger.warning("from_json_handler", extra = msg)
        else:
            self._logger.warning(msg, extra = { "level": "WARNING"} )


    def error(self, msg):
        """
        overrides the logging.error() method
        """
        if isinstance(msg, dict):
            msg["level"] = "ERROR"
            self._logger.error("from_json_handler", extra = msg)
        else:
            self._logger.error(msg, extra = { "level": "ERROR"} )


    def critical(self, msg):
        """
        overrides the logging.critical() method
        """
        if isinstance(msg, dict):
            msg["level"] = "CRITICAL"
            self._logger.critical("from_json_handler", extra = msg)
        else:
            self._logger.critical(msg, extra = { "level": "CRITICAL"} )


    def setLevel(self, level, htype = None ):
        """
        overrides the setLevel method, to deal with level given as a string
        and to change le level of one or all known handlers

        this also change the default level for all future handlers.

        parameter:
        level : level to modify, can be a string or a logging.* level (mandatory)
        htype : if provided (non madatory), change the level of the given handler.
                if not  provided (or None), change the level of all known handlers

        Ex:
            setLevel( logging.DEBUG, 'FILE')
        """

        level = self._internalLevelTranslator(level)
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
            return _x


#
# this is the proper Logger to use
class _FedLogger(_LoggerBase, metaclass=_Singleton):
    pass


logger = _FedLogger()
