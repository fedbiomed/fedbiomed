from typing import List

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import SecaggElementTypes
from fedbiomed.common.message import SecaggReply, NodeMessages


class FakeSecaggSetup:
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            experiment_id: str,
            parties: List[str]):

        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties

        self._message = ''
        self._success = True

    def researcher_id(self) -> str:
        return self._researcher_id

    def secagg_id(self) -> str:
        return self._secagg_id

    def experiment_id(self) -> str:
        return self._experiment_id

    def setup(self) -> SecaggReply:
        return SecaggReply(
            **{
                'researcher_id': self._researcher_id,
                'secagg_id': self._secagg_id,
                'success': self._success,
                'node_id': environ['NODE_ID'],
                'msg': self._message,
            }
        ).get_dict()


class FakeSecaggServkeySetup(FakeSecaggSetup):
    def element(self) -> str:
        return SecaggElementTypes.SERVER_KEY

