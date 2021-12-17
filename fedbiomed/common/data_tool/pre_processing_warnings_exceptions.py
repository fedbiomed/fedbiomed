from enum import Enum
from typing import List


class MissingDataException(Exception):
    def __init__(self,message:str=""):
        self._message = message
        super().__init__(message)
        
    def __str__(self):
        return 'MissingDataException: ' + self._message


class MinimumSamplesViolatedException(Exception):
    def __init__(self, message: str=""):
        self._message = message
        super().__init__(message)
        
    def __str__(self):
        return 'MinimumSamplesViolatedException: ' + self._message


class MissingFeatureException(Exception):
    def __init__(self, message: str = ""):
        self._message = message
        super().__init__(message)
        
    def __str__(self):
        return 'MissingFeatureException: ' + self._message


class MissingViewException(Exception):
    def __init__(self, message: str=""):
        self._message = message
        super().__init__(message)
        
    def __str__(self):
        return 'MissingViewException' + self._message


class DataSanityCheckException(Exception):
    def __init__(self, exceptions: List[Exception]):
        message  = "\n".join([str(exception) for exception in exceptions])
        super().__init__(message)


class WarningType(Enum):
    REGULAR_WARNING = 1
    CRITICAL_WARNING = 2