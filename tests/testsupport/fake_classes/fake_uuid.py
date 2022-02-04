# contains dummy Classes for unit testing, for python library modules

# this avoid re-wirting the same fake classes each time we are desinging a 
# unit test

class FakeUuid:
    VALUE = 1234
    # Fake uuid class
    def __init__(self):
        self.hex = FakeUuid.VALUE

    def __str__(self):
        return str(FakeUuid.VALUE)