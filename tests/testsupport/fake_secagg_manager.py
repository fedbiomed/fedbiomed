from typing import Union, List


class FakeSecaggManager:
    def __init__(self):
        pass


class FakeSecaggServkeyManager(FakeSecaggManager):
    def get(self, secagg_id: str, job_id: str) -> Union[dict, None]:
        return None

    def add(self, secagg_id: str, parties: List[str], job_id: str, servkey_share: str):
        pass

    def remove(self, secagg_id: str, job_id: str) -> bool:
        return True


class FakeSecaggBiprimeManager(FakeSecaggManager):
    def get(self, secagg_id: str) -> Union[dict, None]:
        return None

    def add(self, secagg_id: str, parties: List[str], biprime: str):
        pass

    def remove(self, secagg_id: str) -> bool:
        return True
