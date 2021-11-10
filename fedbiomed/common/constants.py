from enum import Enum


class _BaseEnum(Enum):
    """ Parent class to pass default methods to 
        enumaration classes 
    """

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ComponentType(_BaseEnum):
    """
    Enumeration class, used to characterize the type
    of component of the fedbiomed architecture.
    """

    RESEARCHER = 1
    NODE       = 2

class SecurityLevels(_BaseEnum):

    """ Enumeration class, used to characterize the hashing 
    algorithims
    """

    LOW = 'LOW'
    HIGH = 'HIGH'

