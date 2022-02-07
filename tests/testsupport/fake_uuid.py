""" This file contains dummy Classes for unit testing. It fakes uuid
(from uuid.uuid4()) 
this avoid re-wirting the same fake classes each time we are desinging a 
unit test"""


class FakeUuid:
    """
    Fakes uuid.uuid4(). It is used to return the value
    set by attribute VALUE:
    - when accessing `hex` attribute (eg: replaces uuid.uuid4().hex)
    - when calling `uuid.uuid4()` or `str(uuid.uuid4())`
    
    """
    VALUE = 1234
    def __init__(self):
        """Constructor of the dummy class that fakes uuid
        python standard package.
        Provides `hex` as an attribute, that by default
        always returns `1234`.
        """
        self.hex = FakeUuid.VALUE

    def __str__(self) -> str:
        """Returns attribute VALUE in string 

        Returns:
            str: returns the value set in attribute VALUE.
            By default, returns `'1234'`
        """
        return str(FakeUuid.VALUE)
