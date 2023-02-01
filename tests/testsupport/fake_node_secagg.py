from typing import List

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.message import SecaggReply, NodeMessages


class FakeSecaggSetup:
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            job_id: str,
            sequence: int,
            parties: List[str]):

        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._job_id = job_id
        self._sequence = sequence
        self._parties = parties

        self._message = ''
        self._success = True

    def researcher_id(self) -> str:
        return self._researcher_id

    def secagg_id(self) -> str:
        return self._secagg_id

    def job_id(self) -> str:
        return self._job_id

    def sequence(self) -> str:
        return self._sequence

    def setup(self) -> SecaggReply:
        return NodeMessages.reply_create(
            {
                'researcher_id': self._researcher_id,
                'secagg_id': self._secagg_id,
                'sequence': self._sequence,
                'success': self._success,
                'node_id': environ['NODE_ID'],
                'msg': self._message,
                'command': 'secagg'
            }
        ).get_dict()


class FakeSecaggServkeySetup(FakeSecaggSetup):
    def element(self) -> str:
        return SecaggElementTypes.SERVER_KEY


class FakeSecaggBiprimeSetup(FakeSecaggSetup):
    def element(self) -> str:
        return SecaggElementTypes.BIPRIME
