from fedbiomed.common.component_type import ComponentType
from fedbiomed.common.environ        import Environ

# global dictionnary which contains all environment for the RESEARCHER
environ = Environ(component = ComponentType.RESEARCHER).values()
