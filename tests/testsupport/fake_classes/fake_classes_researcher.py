# contains dummy Classes for unit testing, specifically created for Node
# this avoid re-wirting the same fake classes each time we are desinging a 
# unit test

## Faking FederatedDataSet class

class FederatedDataSetMock():
    def __init__(self, data):
        self._data = data
    def data(self):
        return self._data
