"""
Module that initialize singleton environ object for the node component

[`Environ`][fedbiomed.common.environ] will be initialized after the object `environ`
is imported from `fedbiomed.node.environ`

**Typical use:**

```python
from fedbiomed.node.environ import environ

print(environ['NODE_ID'])
```

"""


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.environ import Environ

# global dictionary which contains all environment for the NODE
environ = Environ(component=ComponentType.NODE)
