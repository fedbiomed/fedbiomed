import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.common.exceptions import FedbiomedNodeStateManagerError
from fedbiomed.node.node_state_manager import NodeStateManager
import testsupport.fake_researcher_environ  ## noqa (remove flake8 false warning)


class TestNodeStateManager(unittest.TestCase):
    
    
    def setUp(self) -> None:
        self.query_patcher = patch('fedbiomed.node.node_state_manager.Query')
        self.table_patcher = patch('fedbiomed.node.node_state_manager.TinyDB')

        self.query_patcher.start()
        self.table_patcher.start()
        self.test_nsm = NodeStateManager("path/to/db")
    
    def tearDown(self) -> None:
        self.query_patcher.stop()
        self.table_patcher.stop()
    
    def test_node_state_manager_1_fail_to_build(self):
        query_patcher = patch('fedbiomed.node.node_state_manager.Query')
        table_mock = MagicMock(table=MagicMock(side_effect=NameError("this is a test")))
        db_patcher = patch('fedbiomed.node.node_state_manager.TinyDB', return_value = table_mock)
        table_patcher = patch('fedbiomed.node.node_state_manager.TinyDB.table', side_effect=NameError("this is a test"))
        
        
        query_patcher.start()
        table_patcher.start()
        db_patcher.start()
        
        with self.assertRaises(FedbiomedNodeStateManagerError):
            nsm = NodeStateManager('path/to/db')
        
        query_patcher.stop()
        table_patcher.stop()
        db_patcher.stop()
    
    
    @patch('fedbiomed.node.node_state_manager.raise_for_version_compatibility')
    def test_node_state_manager_2_get(self, raise_for_compatibility_patch):
        
        #self.query_patcher = patch('fedbiomed.node.node_state_manager.Query')
        
        #self.
        test_nsm = self.test_nsm
        test_nsm.get('job_id', 'state_id')
        
        
        raise_for_compatibility_patch.assert_called_once()
        

if __name__ == '__main__':  # pragma: no cover
    unittest.main()