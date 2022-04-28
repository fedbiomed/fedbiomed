'''
Create the environ singleton fot the node component.
'''


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.environ   import Environ

# global dictionary which contains all environment for the NODE
environ = Environ(component = ComponentType.NODE)
