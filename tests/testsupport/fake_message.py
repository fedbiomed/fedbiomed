# Fakes NodeMessage (from fedbiomed.common.messages)
from typing import Any, Dict


class FakeNodeMessages:
    # Fake NodeMessage
    def __init__(self, msg: Dict[str, Any]):
        self.msg = msg

    def get_dict(self) -> Dict[str, Any]:
        return self.msg