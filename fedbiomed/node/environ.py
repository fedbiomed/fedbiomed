from fedbiomed.common.component_type import ComponentType
from fedbiomed.common.environ        import Environ

# global dictionnary which contains all environment for the NODE
environ = Environ(component = ComponentType.NODE).values()
