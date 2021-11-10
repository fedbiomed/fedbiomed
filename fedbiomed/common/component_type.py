from enum import Enum


class ComponentType(Enum):
    """
    Enumeration class, used to characterize the type
    of component of the fedbiomed architecture.
    """
    RESEARCHER = 1
    NODE       = 2
