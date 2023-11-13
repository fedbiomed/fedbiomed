import unittest

from unittest.mock import MagicMock
from fedbiomed.common.exceptions import FedbiomedNodeStateAgentError
from fedbiomed.researcher.datasets import FederatedDataSet

from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.responses import Responses


class TestNodeStateAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.node_ids_1 = ['node_id_1234',
                           'node_id_5678',
                           'node_id_9012']

    def test_node_state_1_agent_get_last_node_states(self):

        # build several NodeStateAgents
        # case NodeStateAgent is built with no nodes
        test_nsa_1 = NodeStateAgent([])
        res_1 = test_nsa_1.get_last_node_states()

        self.assertEqual(res_1, {})

        # case NodeStateAgent is built with some nodes

        node_ids = ['node_id_1234',
                    'node_id_5678',
                    'node_id_9012']
        test_nsa_2 = NodeStateAgent(node_ids)
        res_2 = test_nsa_2.get_last_node_states()

        expected_res = {k: None for k in node_ids}
        self.assertDictEqual(expected_res, res_2)

    def test_node_state_agent_4_upate_node_state(self):
        node_ids_2 = ['node_id_1234',
                      'node_id_5678',
                      'node_id_4321',
                      'node_id_0987']

        nsa = NodeStateAgent(self.node_ids_1)
        # case where Responses is None

        nsa.update_node_states(node_ids_2)

        res = nsa.get_last_node_states()

        expected_res = {k: None for k in node_ids_2}
        self.assertDictEqual(res, expected_res)  # we check here that keys and initialization has been done

        # now we update wrt Responses

        resp = {
            'node_id_1234': {'node_id': 'node_id_1234',
                             'state_id': 'node_state_1234'},
            'node_id_5678': {'node_id': 'node_id_5678',
                             'state_id': 'node_state_5678'}
        }

        nsa.update_node_states(node_ids_2, resp)
        res = nsa.get_last_node_states()
        self.assertListEqual(list(res.keys()), node_ids_2)

        # finally, we update with a node_id that is not present in the FederatedDataset

        nodes_replies_content = {
            'node_id_1234': {'node_id': 'node_id_1234',
                                  'state_id': 'node_state_1234'},
            'node_id_5678': {'node_id': 'node_id_5678',
                             'state_id': 'node_state_5678'},
            'unknown-node_id': {'node_id': 'unknown-node_id',
                                'state_id': 'unknown_state-id'} 
        }
        resp = nodes_replies_content

        nsa.update_node_states(node_ids_2, resp)
        res = nsa.get_last_node_states()

        nodes_replies_content_nodes_id = list(nodes_replies_content.keys())
        nodes_replies_content_nodes_id.remove('unknown-node_id')
        self.assertListEqual(list(res.keys()), node_ids_2)
        self.assertNotIn('unknown-node_id', res)
        for node_id in nodes_replies_content_nodes_id:
            self.assertIn(node_id, node_ids_2)

    def test_node_state_agent_5_update_node_state_failure(self):
        nsa = NodeStateAgent(self.node_ids_1)
        
        bad_resp = {'node_id_1234': {'node_id': 'node_id_1234',
                                     'state_id': 'node_state_1234'},
                    'node_id_5678': {'node_id': 'node_id_5678'}
                    }

        with self.assertRaises(FedbiomedNodeStateAgentError):
            nsa.update_node_states(self.node_ids_1, bad_resp)

    def test_node_state_agent_6_save_state_breakpoint(self):
        nsa = NodeStateAgent(self.node_ids_1)

        nsa_bkpt = nsa.save_state_breakpoint()
        res = nsa.get_last_node_states()

        self.assertDictEqual(nsa_bkpt['collection_state_ids'], res)

    def test_node_state_agent_7_load_state_breakpoint(self):
        nodes_states_bkpt = {
            'collection_state_ids': {
                'node_id_1234': 'state_id_1234',
                'node_id_5678': 'state_id_5678',
                'node_id_9012': 'state_id_9012'
            },
        }

        nsa = NodeStateAgent(self.node_ids_1)
        nsa.load_state_breakpoint(nodes_states_bkpt)

        reloaded_nodes_states_bkpt = nsa.save_state_breakpoint()

        self.assertDictEqual(nodes_states_bkpt, reloaded_nodes_states_bkpt)

    def test_node_state_agent_8_save_and_load_bkpt(self):
        nsa = NodeStateAgent(self.node_ids_1)
        last_nodes_states_before_saving = nsa.get_last_node_states()

        nodes_states_bkpt = nsa.save_state_breakpoint()
        nsa2 = NodeStateAgent(self.node_ids_1)
        nsa2.load_state_breakpoint(nodes_states_bkpt)

        last_nodes_states_after_saving = nsa2.get_last_node_states()

        self.assertDictEqual(last_nodes_states_after_saving, last_nodes_states_before_saving)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
