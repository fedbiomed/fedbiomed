import paho.mqtt.publish as publish

import logging
import logging.handlers

import json_log_formatter

# default values
DEFAULT_LOGFILE = 'mylog.log'
DEFAULT_LEVEL   = logging.WARNING
DEFAULT_TOPIC   = 'general/logger'

#
# mqtt handler
#
class MqttHandler(logging.Handler):
    """
    A handler class to deal with MQTT
    """

    def __init__(self,
                 client_id   = None,
                 hostname    = None,
                 port        = None,
                 topic       = DEFAULT_TOPIC
                 ):
        """
        Constructor

        parameters:
        client_id : unique MQTTT client id
        hostname  : MQTT server hostname
        port      : MQTT port
        topic     : topic/channel to publish to
        """

        logging.Handler.__init__(self)
        self._client_id = client_id
        self._hostname  = hostname
        self._port      = port
        self._topic     = topic

    def emit(self, record):
        """
        do the proper job (override the logging.Handler method() )

        parameters:
        record : is automatically passed by the logger class
        """

        msg = self.format(record)
        publish.single(self._topic,
                       msg,
                       hostname  = self._hostname,
                       port      = self._port,
                       client_id = self._client_id)


#
# singletonizer: transforms a class to a singleton
# nothin else to say really !
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


    def __init__(self, level = DEFAULT_LEVEL ):
        """
        constructor of base class

        parameter: initial loglevel.
        This loglevel will be the default for all handlers, if called
        without the default level

        an initial console logger is installed (so the logger has at minimum one handler)
        """

        # name this logger
        self._logger = logging.getLogger("fedbiomed")

        self._default_level = DEFAULT_LEVEL  # MANDATORY ! KEEP THIS PLEASE !!!
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


    def _internalLevelTranslator(self, level = DEFAULT_LEVEL) :
        """
        this helper allows to use a string instead of logging.* then using logger levels

        parameter:
        level:  wanted parameter

        output:
        level:  logging.* form of the level

        ex:
        _internalLevelTranslator('DEBUG')  returns logging.DEBUG
        """

        # transform string to logging.level
        _levels = {
            "DEBUG"          : logging.DEBUG,
            "INFO"           : logging.INFO,
            "WARNING"        : logging.WARNING,
            "ERROR"          : logging.ERROR,
            "CRITICAL"       : logging.CRITICAL,
        }

        # to ease the validation test
        _original_levels = {
            logging.DEBUG    : None,
            logging.INFO     : None,
            logging.WARNING  : None,
            logging.ERROR    : None,
            logging.CRITICAL : None
        }

        # logging.*
        if level in _original_levels:
            return level

        # strings
        if isinstance(level, str):
            upperlevel = level.upper()

            if upperlevel in _levels:
                return _levels[upperlevel]

        # bad input !

        # should always work, even at startup with _logger badly initialized
        # because this method is called by __init__
        # (where else to log this really ?)
        self._logger.warning("calling selLevel() with bad value: " + str(level))
        return DEFAULT_LEVEL



    def addJsonFileHandler(self, filename = DEFAULT_LOGFILE, level = DEFAULT_LEVEL):
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
                          level  = DEFAULT_LEVEL):

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
                       client_id   = None,
                       hostname    = None,
                       port        = None,
                       topic       = DEFAULT_TOPIC,
                       level       = DEFAULT_LEVEL
                       ):

        """
        add a mqqt handler, to publish error message on a topic

        parameters:
        client_id   : unique client id of the caller
        hostname    : MQTT server hostname
        port        : MQTT port
        topic       : topic to publish to    (non mandatory)
        level       : level of this handler  (non mandatory)
        """
        handler = MqttHandler( client_id   = client_id ,
                               hostname    = hostname,
                               port        = port,
                               topic       = topic
                              )

        handler.setLevel( self._internalLevelTranslator(level) )
        formatter = json_log_formatter.JSONFormatter()
        # old oneline format
        #formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

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
            self._logger.log(level, msg, extra = { "level": "DEBUG"} )


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
        self._logger.warning(htype, "handler not initialized yet")


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
