from typing import Union, List


class FakeSecaggManager:
    def __init__(self):
        pass


class FakeSecaggServkeyManager(FakeSecaggManager):
    def get(self, secagg_id: str, experiment_id: str) -> Union[dict, None]:
        return None

    def add(
        self, secagg_id: str, parties: List[str], experiment_id: str, servkey_share: str
    ):
        pass

    def remove(self, secagg_id: str, experiment_id: str) -> bool:
        return True
