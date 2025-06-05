class FakeTinyDB:
    def __init__(self, path):
        self._table = FakeTable()

    def table(self, *args, **kwargs):
        return self._table


class FakeQuery:
    def __init__(self):
        class FakeSecaggId:
            def __init__(self):
                self.exists_value = True

            def exists(self):
                return self.exists_value

            def one_of(self, id):
                return True
        class FakeDistantNodeId:
            def __init__(self):
                self.exists_value = True

            def exists(self):
                return self.exists_value

        self.secagg_id = FakeSecaggId()
        self.distant_node_id = FakeDistantNodeId()

        class Type():
            def exists(self):
                return True
        self.type = Type()


class FakeTable:
    def __init__(self):
        self.entries = []
        self.exception_list = False
        self.exception_get = False
        self.exception_insert = False
        self.exception_upsert = False
        self.exception_search = False
        self.exception_remove = False

    def all(self):
        if self.exception_list:
            raise Exception('mocked exception')
        else:
            return self.entries

    def get(self, *args, **kwargs):
        if self.exception_get:
            raise Exception('mocked exception')
        else:
            # super simplified
            return self.entries[0]

    def insert(self, entry):
        if self.exception_insert:
            raise Exception('mocked exception')
        else:
            self.entries.append(entry)

    def upsert(self, entry, condition):
        if self.exception_upsert:
            raise Exception('mocked exception')
        else:
            self.entries.append(entry)


    def search(self, *args, **kwargs):
        if self.exception_search:
            raise Exception('mocked exception')
        else:
            return self.entries

    def remove(self, *args, **kwargs):
        if self.exception_remove:
            raise Exception('mocked exception')
        else:
            self.entries = []
            return True
