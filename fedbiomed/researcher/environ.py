"""
Module that initialize singleton environ object for the researcher component

[`Environ`][fedbiomed.common.environ] will be initialized after the object `environ`
is imported from `fedbiomed.researcher.environ`

**Typical use:**

```python
from fedbiomed.researcher.environ import environ

print(environ['RESEARCHER_ID'])
```

"""


from fedbiomed.common.constants import ComponentType
from fedbiomed.common.environ import Environ

# Global dictionary which contains all environment for the RESEARCHER
environ = Environ(component=ComponentType.RESEARCHER)
