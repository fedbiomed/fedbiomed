import unittest

from typing import Any, Dict
from unittest.mock import MagicMock, call, create_autospec, patch
from fedbiomed.common.exceptions import FedBiomedNodeStateAgentError
from fedbiomed.researcher.datasets import FederatedDataSet

from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.responses import Responses


class TestNodeStateAgent(unittest.TestCase):
    
    def setUp(self) -> None:
        self.fds_data_1 = {'node_id_1234': MagicMock(),
                            'node_id_5678': MagicMock(),
                            'node_id_9012': MagicMock()}

    def test_node_state_1_agent_get_last_node_states(self):
        
        # build several NodeStateAgents
        # case NodeStateAgent is built without arguments
        test_nsa_1 = NodeStateAgent()
        res_1 = test_nsa_1.get_last_node_states()
        
        self.assertIsNone(res_1)

        # case NodeStateAgent is built with dictionary
        
        fds_data = {'node_id_1234': MagicMock(),
                    'node_id_5678': MagicMock(),
                    'node_id_9012': MagicMock()}
        test_nsa_2 = NodeStateAgent(fds_data)
        res_2 = test_nsa_2.get_last_node_states()
        
        expected_res = {k: None for k in fds_data}
        self.assertDictEqual(expected_res, res_2)
        
        
        # case NodeStateAgent is built with FederatedDataset
        
        fds = MagicMock(spec=FederatedDataSet)
        fds.data.return_value = fds_data
        
        test_nsa_3 = NodeStateAgent(fds)
        res_3 = test_nsa_3.get_last_node_states()
        
        self.assertDictEqual(res_3, expected_res)


    def test_node_state_agent_2_set_federated_dataset(self):
        # test with several way of creating NodeStateAgent
        fds_data_1 = {'node_id_1234': MagicMock(),
                      'node_id_5678': MagicMock(),
                      'node_id_9012': MagicMock()}
        
        fds_data_2 = MagicMock(spec=FederatedDataSet)
        
        
        for fds_data in (fds_data_1, fds_data_2, ):
            test_nsa = NodeStateAgent(fds_data)
            
            fds_data_new = MagicMock(spec=FederatedDataSet)
            fds_data_new.data = MagicMock(spec=dict)

            test_nsa.set_federated_dataset(fds_data_new)
            
            fds_data_new.data.assert_called_once()
            
    
    def test_node_state_agent_3_raise_error(self):
        fds_data_1 = {'node_id_1234': MagicMock(),
                      'node_id_5678': MagicMock(),
                      'node_id_9012': MagicMock()}
        
        fds_data_2 = MagicMock(spec=FederatedDataSet)
        
        
        for fds_data in (fds_data_1, fds_data_2, ):
            test_nsa = NodeStateAgent(fds_data)
            
            with self.assertRaises(FedBiomedNodeStateAgentError):
                test_nsa.set_federated_dataset(MagicMock())

    def test_node_state_agent_4_upate_node_state(self):
        fds = MagicMock(spec=FederatedDataSet, data=MagicMock(return_value=self.fds_data_1))
        fds_data_2 = {'node_id_1234': MagicMock(),
                    'node_id_5678': MagicMock(),
                    'node_id_4321': MagicMock(),
                    'node_id_0987': MagicMock()}

        nsa = NodeStateAgent(fds)
        
        # case where Responses is None
        
        nsa.update_node_states(fds_data_2)
        
        res = nsa.get_last_node_states()
        
        all_node_ids = list(fds_data_2.keys())
        all_node_ids.extend(list(self.fds_data_1))
        expected_res = {k: None for k in all_node_ids}
        self.assertDictEqual(res, expected_res)
        
        # now we update wrt Responses
        
        # TODO: to be continued...
        
        resp = Responses([{'node_id': 'node_id_1234',
                           'state_id': 'node_state_1234'}, 
                          {'node_id': 'node_id_5678',
                           'state_id': 'node_state_5678'}
                          ])
        print("RESPONSE", resp)
        
        nsa.update_node_states(fds_data_2, resp)
        res = nsa.get_last_node_states()
        self.assertIn(list(res.keys()), list(fds_data_2.keys()))
        
        # finally, we update with a node_id that is not present in the FederatedDataset
        
        resp = Responses([{'node_id': 'node_id_1234',
                           'state_id': 'node_state_1234'}, 
                          {'node_id': 'node_id_5678',
                           'state_id': 'node_state_5678'},
                           {'node_id': 'unknown-node_id', 
                            'state_id': 'unknown_state-id'}
                          ])
        
    def test_node_state_agent_5_save_state_ids_in_bkpt(self):
        pass
    
    def test_node_state_agent_6_loa_state_ids_in_bkpt(self):
        pass
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
