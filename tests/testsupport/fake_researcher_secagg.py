from typing import Union, List

FAKE_CONTEXT_VALUE = "MY_CONTEXT"

class FakeSecaggContext:
    def __init__(self, parties: List[str]):
        self.parties = parties
        self.stat = False
        self.cont = None

    def status(self) -> bool:
        return self.stat

    def context(self) -> Union[dict, None]:
        return self.cont

    def setup(self, timeout: float = 0) -> bool:
        self.stat = True
        self.cont = FAKE_CONTEXT_VALUE
        return True


class FakeSecaggServkeyContext(FakeSecaggContext):
    pass


class FakeSecaggBiprimeContext(FakeSecaggContext):
    pass
