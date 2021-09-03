import logging
import logging.handlers

import json_log_formatter

#
LOGFILE = 'mylog.log'

# transform string to logging.level
_levels = {
    "DEBUG"          : logging.DEBUG,
    "INFO"           : logging.INFO,
    "WARNING"        : logging.WARNING,
    "ERROR"          : logging.ERROR,
    "CRITICAL"       : logging.CRITICAL,
}

# to ease a validation test
_original_levels = {
    logging.DEBUG    : None,
    logging.INFO     : None,
    logging.WARNING  : None,
    logging.ERROR    : None,
    logging.CRITICAL : None
}

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


    def __init__(self):

        self._logger = logging.getLogger("fedbiomed")
        fhandler  = logging.FileHandler(filename=LOGFILE, mode='a')

        formatter = json_log_formatter.JSONFormatter()

        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        self._logger.addHandler(fhandler)

        # log level
        self._logger.setLevel(logging.DEBUG)

        pass


    def debug(self, msg):
        if isinstance(msg, dict):
            msg["level"] = "DEBUG"
            self._logger.debug("from_json_handler", extra = msg)
        else:
            self._logger.debug(msg, extra = { "level": "DEBUG"} )

    def info(self, msg):
        if isinstance(msg, dict):
            msg["level"] = "INFO"
            self._logger.info("from_json_handler", extra = msg)
        else:
            self._logger.info(msg, extra = { "level": "INFO"} )

    def warning(self, msg):
        if isinstance(msg, dict):
            msg["level"] = "WARNING"
            self._logger.warning("from_json_handler", extra = msg)
        else:
            self._logger.warning(msg, extra = { "level": "WARNING"} )

    def error(self, msg):
        if isinstance(msg, dict):
            msg["level"] = "ERROR"
            self._logger.error("from_json_handler", extra = msg)
        else:
            self._logger.error(msg, extra = { "level": "ERROR"} )

    def critical(self, msg):
        if isinstance(msg, dict):
            msg["level"] = "CRITICAL"
            self._logger.critical("from_json_handler", extra = msg)
        else:
            self._logger.critical(msg, extra = { "level": "CRITICAL"} )


    def setLevel(self, level = None):
        #
        # use original logging.levels (need to import logging before)
        if level in _original_levels:
            print("setting level to", str(level))
            self._logger.setLevel(level)
            return

        #
        # use strings as debug levels
        if isinstance(level, str):
            level = level.upper()

        if level in _levels:
            self._logger.setLevel(_levels[level])
        else:
            # TODO: where to log this really ?
            self._logger.debug("calling selLevel() with bad value")


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
