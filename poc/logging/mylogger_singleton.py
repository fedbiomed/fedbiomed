import logging
import logging.handlers

import json_log_formatter

LOGFILE = 'mylog.log'
#
# singletonizer: transforms a class to a sigleton
class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


#
# this is the base class to use, the proper Logger
class _LoggerBase():
    def __init__(self):
        self._logger = logging.getLogger()
        fhandler  = logging.FileHandler(filename=LOGFILE, mode='a')

        formatter = json_log_formatter.JSONFormatter()

        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        self._logger.addHandler(fhandler)

        # rotation handler
        #rhandler = logging.handlers.RotatingFileHandler(LOGFILE,
        #                                                maxBytes=(1024*1024),
        #                                                backupCount=10 )
        #self._logger.addHandler(rhandler)


        # log level
        self._logger.setLevel(logging.DEBUG)

        pass

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg, extra = {"titi": "toto"} )

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)


#
# this is the proper Logger to use
class _MyLogger(_LoggerBase, metaclass=_Singleton):
    pass


logger = _MyLogger()
