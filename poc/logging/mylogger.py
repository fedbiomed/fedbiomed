import paho.mqtt.publish as publish

import logging
import logging.handlers

import json_log_formatter

#
LOGFILE = 'mylog.log'

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
                 channel     = "logger"
                 ):

        logging.Handler.__init__(self)
        self._client_id = client_id
        self._hostname  = hostname
        self._port      = port
        self._channel   = channel

    def emit(self, record):
        msg = self.format(record)
        #self._mqtt.publish(self._channel,msg)
        publish.single(self._channel,
                       msg,
                       hostname  = self._hostname,
                       port      = self._port,
                       client_id = self._client_id)


#
# singletonizer: transforms a class to a sigleton
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
    the _logger member og the class if necessary (instead of overloading it)
    (ex:  logger._logger.getEffectiveLevel() )
    """


    def __init__(self, level = logging.DEBUG ):
        """
        constructor of base class

        parameter: initial loglevel.
        This loglevel will be the default for all handlers.

        initial console logger is installed.
        """

        self._logger = logging.getLogger("fedbiomed")
        self._handlers = {}

        # this will be used by the _levelTranslator method
        self._default_level = logging.DEBUG

        self._default_level = self._internalLevelTranslator(level)
        self._logger.setLevel(self._default_level)

        # add a console handler on startup
        self.addConsoleHandler()

        pass


    def _internalAddHandler(self, output , handler):
        """
        private method

        add a handler to the logger. only one handler is allowed
        for a given output (type)

        parameters:
        output  = tag for the logger ("CONSOLE", "FILE")
        handler = proper handler to install
        """
        if not handler in self._handlers:
            self._handlers[output] = handler
            self._logger.addHandler(handler)
        else:
            self._logger.debug(output, "handler already present")

        pass


    def _internalLevelTranslator(self, level) :
        """
        allows to use string instead of logging.* then using logger levels
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
            level = level.upper()

        if level in _levels:
            return _levels[level]
        else:
            # TODO: where to log this really ?
            self._logger.debug("calling selLevel() with bad value")
            return self._default_level

        pass

    def addJsonFileHandler(self, filename = LOGFILE, level = logging.DEBUG):
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
                          level  = logging.DEBUG):

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
                       channel     = "logger",
                       level       = logging.DEBUG
                       ):

        """
        add a mqqt handler, to publish error message on a channel

        parameters:
        mqqt: mandatory mqtt connector
        channel: channel to pubish on, defaut "error"
        level  : initial level of the logger for this handler (optionnal)
                 if not given, the default level is set
        """
        handler = MqttHandler( client_id   = client_id ,
                               hostname    = hostname,
                               port        = port,
                               channel     = channel
                              )

        handler.setLevel( self._internalLevelTranslator(level) )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

        handler.setFormatter(formatter)
        self._internalAddHandler("MQTT", handler)

        pass


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


    def setLevel(self, level ):
        """
        overrides the setLevel method, to deal with level given as a string
        and to change le level of all handler
        """

        level = self._internalLevelTranslator(level)
        self._logger.setLevel( level )
        for h in self._handlers:
            self._handlers[h].setLevel(level)


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
class _MyLogger(_LoggerBase, metaclass=_Singleton):
    pass


logger = _MyLogger()
