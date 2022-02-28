'''
create the environ singleton fot the researcher component
'''


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.environ   import Environ

# global dictionnary which contains all environment for the RESEARCHER
environ = Environ(component = ComponentType.RESEARCHER)
