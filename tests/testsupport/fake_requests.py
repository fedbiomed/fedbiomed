from typing import Union

from testsupport.fake_responses import FakeResponses


class FakeRequests():
    def __init__(self):
        self.messages = []
        self.sequence = 0

    def send_message_side_effect(self, msg: dict, client: str = None, add_sequence: bool = False) -> \
            Union[int, None]:
        # always add sequence, whatever the `add_sequence`
        self.sequence += 1
        if msg['command'] in ['secagg', 'secagg-delete']:
            message = {
                'researcher_id': msg['researcher_id'],
                'secagg_id': msg['secagg_id'],
                'sequence': self.sequence,
                'success': True,
                'node_id': client,
                'msg': 'Fake request',
                'command': msg['command'],
            }
        else:
            message = {}
        self.messages.append(message)
        return self.sequence

    def get_responses_side_effect(
            self,
            look_for_commands: list,
            timeout: float = None,
            only_successful: bool = True,
            while_responses: bool = True) -> FakeResponses:
        # return existing responses without delay, whatever the arguments
        messages = self.messages
        self.messages = []
        return FakeResponses(messages)
