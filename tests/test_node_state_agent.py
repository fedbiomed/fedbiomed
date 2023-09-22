import unittest

from typing import Any, Dict
from unittest.mock import MagicMock, call, create_autospec, patch
from fedbiomed.researcher.datasets import FederatedDataSet

from fedbiomed.researcher.node_state_agent import NodeStateAgent


class TestNodeStateAgent(unittest.TestCase):
    def test_1_get_last_node_states(self):
        
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
