import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.common.exceptions import FedbiomedNodeStateManagerError
from fedbiomed.node.node_state_manager import NodeStateManager, NodeStateFileName
import testsupport.fake_researcher_environ  ## noqa (remove flake8 false warning)


class TestNodeStateManager(unittest.TestCase):
    
    
    def setUp(self) -> None:
        self.query_patcher = patch('fedbiomed.node.node_state_manager.Query')
        self.table_patcher = patch('fedbiomed.node.node_state_manager.TinyDB')

        self.query_mock = self.query_patcher.start()
        self.table_mock = self.table_patcher.start()
        self.test_nsm = NodeStateManager("path/to/db")
    
    def tearDown(self) -> None:
        self.query_patcher.stop()
        self.table_patcher.stop()
    
    def test_node_state_manager_1_fail_to_build(self):
        query_patcher = patch('fedbiomed.node.node_state_manager.Query')
        table_mock = MagicMock(table=MagicMock(side_effect=NameError("this is a test")))
        db_patcher = patch('fedbiomed.node.node_state_manager.TinyDB', return_value = table_mock)
        table_patcher = patch('fedbiomed.node.node_state_manager.TinyDB.table')
        
        
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
        job_id , state_id = 'job_id' , 'state_id'
        self.query_mock.job_id = MagicMock(return_value=job_id)
        self.query_mock.state_id = MagicMock(return_value=state_id)
        self.table_mock.return_value.table.return_value.get.return_value = {
            "version_node_id" : '1.2.3',
            'state_id': state_id,
            'job_id': job_id,
            'state': {}
        }
        test_nsm = self.test_nsm
        res = test_nsm.get(job_id, state_id)

        self.table_mock.return_value.table.return_value.get.assert_called_once_with(
            (self.query_mock.job_id == job_id) & (self.query_mock.state_id == state_id)
        )
        self.assertIsInstance(res, dict)
        raise_for_compatibility_patch.assert_called_once()
    
    @patch('fedbiomed.node.node_state_manager.NodeStateManager._load_state') 
    def test_node_state_manager_3_get_error(self, private_load_state_mock):
        private_load_state_mock.return_value = None
        job_id , state_id = 'job_id' , 'state_id'
        with self.assertRaises(FedbiomedNodeStateManagerError):
            self.test_nsm.get(job_id, state_id) 
        

    def test_node_state_manager_4_add(self):
        fake_state = {}
        self.table_mock.return_value.table = MagicMock(upsert=fake_state)
        # TODO: to be finished
        

class TestNodeStateFileName(unittest.TestCase):
    def test_node_state_file_name_1_correct_entries(self):
        # here we test that all entries of NodeStateFIleName enum class respect convention
        
        
        for entry_value in NodeStateFileName.list():
            try:
                entry_value % ('string_1', 'string_2')
            except TypeError as te:
                self.assertTrue(False, f"error in NodeStateFileName, entry {entry_value} doesnot respect convention"
                                f" details : {te}") 


if __name__ == '__main__':  # pragma: no cover
    unittest.main()