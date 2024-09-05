import unittest
from unittest.mock import MagicMock, patch
import uuid
from fedbiomed.common.secagg import Secret, Share
from fedbiomed.common.constants import REQUEST_PREFIX, ErrorNumbers, __messaging_protocol_version__
from fedbiomed.common.message import Message, NodeMessages, NodeToNodeMessages, ResearcherMessages
from testsupport.fake_uuid import FakeUuid

# Assuming the classes Secret and Share are defined in secret_module
# from secret_module import Secret, Share


# TODO: move these tests into `test_message`
class TestSecret(unittest.TestCase):

    def setUp(self) -> None:
        self.secrets = (Secret(123),
                        Secret([1,2,3]),
                        Secret([1]))
        

    @patch('uuid.uuid4')
    def test_01_n2n_messages(self, uuid_patch):
        
        uuid_patch.return_value = FakeUuid()
        for secret in self.secrets:
            shares = secret.split(3)
            outgoing_msg = NodeToNodeMessages.format_outgoing_message({
                                'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                                'node_id': '1234',
                                'dest_node_id': 'dataset_node_1234',
                                'secagg_id': 'secagg_id_1234',
                                'command': 'additive-secret-share-request',
                                'public_key': bytes(1234),
                                #'share': secret.secret
                            })

            self.assertDictEqual(outgoing_msg.get_dict(),
                                 {
                                     'protocol_version': str(__messaging_protocol_version__),
                                     'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                                     'node_id': '1234',
                                     'dest_node_id': 'dataset_node_1234',
                                     'secagg_id': 'secagg_id_1234',
                                     'command': 'additive-secret-share-request',
                                     'public_key': bytes(1234),})

            for share in shares:
                ingoing_msg = NodeToNodeMessages.format_outgoing_message({
                                    'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                                    'node_id': '1234',
                                    'dest_node_id': 'dataset_node_1234',
                                    'secagg_id': 'secagg_id_1234',
                                    'command': 'additive-secret-share-reply',
                                    'public_key': bytes(1234),
                                    'share': share.value
                                })

                self.assertDictEqual(ingoing_msg.get_dict(),
                                     {
                                         'protocol_version': str(__messaging_protocol_version__),
                                         'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                                         'node_id': '1234',
                                         'dest_node_id': 'dataset_node_1234',
                                         'secagg_id': 'secagg_id_1234',
                                         'command': 'additive-secret-share-reply',
                                         'public_key': bytes(1234),
                                         'share': share.value
                                     })

    @patch('uuid.uuid4')   
    def test_02_request_reply_messages(self, uuid_patch):

        uuid_patch.return_value = FakeUuid()
        outgoing_msg = ResearcherMessages.format_outgoing_message(
                    {
                     'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                     'command': 'secagg-additive-ss-setup-request',
                     'element': 3,
                     'researcher_id': 'researcher_1234',
                     'experiment_id': 'exp_1234',
                     'parties': ['node_1', 'node_2'],
                     'secagg_id': 'secagg_1234'
 
                    })
        
        for secret in self.secrets:
            shares = secret.split(3)
            for share in shares:

                outgoing_msg_1 = NodeMessages.format_outgoing_message({
                    'success': True,
                     'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                     'command': 'secagg-additive-ss-setup-reply',
                     'researcher_id': 'researcher_1234',
                     'secagg_id': 'secagg_1234',  
                     'node_id': 'node_id_1234',
                     'msg': 'test',
                     'share': share.value
                    })

                self.assertDictEqual(outgoing_msg_1.get_dict(),
                                     {
                                         'protocol_version': str(__messaging_protocol_version__),
                                         'success': True,
                                         'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                                         'command': 'secagg-additive-ss-setup-reply',
                                         'researcher_id': 'researcher_1234',
                                         'secagg_id': 'secagg_1234',  
                                         'node_id': 'node_id_1234',
                                         'msg': 'test',
                                         'share': share.value   
 
                                      }
                                     )
