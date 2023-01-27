import sys

from unittest.mock import Mock
from fedbiomed.common.environ import Environ


class Fake(object):
    @classmethod
    def imitate(cls, class_):
        for name in class_.__dict__:
            try:
                setattr(cls, name, Mock())
            except (TypeError, AttributeError):
                pass
        return cls


EnvironMock = Fake.imitate(Environ)


class Environ(EnvironMock):
    def __init__(self, root_dir):
        self._values = {}


# class FakeEnvironModule:
#     Environ = Environ


# sys.modules["fedbiomed.common.environ"] = FakeEnvironModule