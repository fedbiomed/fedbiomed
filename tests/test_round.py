# Managing NODE, RESEARCHER environ mock before running tests
from typing import Any, Dict, List
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.node.environ import environ

import time

import unittest
from unittest.mock import MagicMock, patch

from fedbiomed.node.round import Round


class TestRound(unittest.TestCase):
    SLEEPING_TIME = 1  # time that simulate training (in seconds)
    class FakeModel:
        # Fake model that mimics a Training Plan model
        def __init__(self, **kwargs):
            pass
        def load(self, path:str, to_params:bool):
            pass
        def save(self, filename:str, results: Dict[str, Any]):
            pass
        def set_dataset(self, path:str):
            pass
        def training_routine(self, **kwargs):
            time.sleep(TestRound.SLEEPING_TIME)
        def after_training_params(self)-> List[int]:
            return [1, 2, 3, 4]
        
            
    @patch('fedbiomed.common.repository.Repository.__init__')
    @patch('fedbiomed.node.model_manager.ModelManager.__init__')
    def setUp(self,
              model_manager_patch,
              reporistory_patch):
        model_manager_patch.return_value = None
        reporistory_patch.return_value = None
        
        # instanciate Round class
        self.r1 = Round(model_url='http://somewhere/where/my/model?is_stored=True',
                        model_class='my_training_plan')
        
    
    def tearDown(self) -> None:
        pass
    
    def _init_normal_case_scenario(self):
        pass
    
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_01(self, 
                                   uuid_patch,
                                   repository_download_patch,
                                   model_manager_patch,
                                   builtin_exec_patch,
                                   builtin_eval_patch,
                                   repository_upload_patch,
                                   node_msg_patch
                                   ):
        """tests correct execution and message parameters. 
        Besides test tests the training time.
        """
        TestRound.SLEEPING_TIME = 1
        URL_MSG = 'http://url/where/my/file?is=True'
        # initalisation of dummy classes
        class FakeUuid:
            # Fake uuid class
            def __init__(self):
                self.hex = 1234
                
            def __str__(self):
                return '1234'
            
        class FakeMonitor:
            def history(self) -> Dict[str, float]:
                return {'Loss': 0.1}
            
        class FakeNodeMessages:
            def __init__(self, msg: Dict[str, Any]):
                self.msg = msg

            def get_dict(self) -> Dict[str, Any]:
                return self.msg

        def repository_side_effect(model_url: str, model_name: str):
            return 200, 'my_python_model'
        
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeNodeMessages(msg)
            return fake_node_msg

        # initialisation of patchers 
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        model_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = self.FakeModel
        repository_upload_patch.return_value = {'file': URL_MSG}
        node_msg_patch.side_effect = node_msg_side_effect
        
        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}
        self.r1.monitor = FakeMonitor()
        
        # action!
        msg = self.r1.run_model_training()
        
        # check results
        self.assertTrue(msg.get('success', False))
        self.assertEqual(URL_MSG, msg.get('params_url'))
        self.assertEqual('train', msg.get('command'))
        
        # timing test
        if not isinstance(msg, dict) or not hasattr(msg, 'get'):
            self.skipTest('timing tests skipped because of incorrect data type returned')
        self.assertAlmostEqual(
            msg.get('timing', {'rtime_training': 0}).get('rtime_training'),
            TestRound.SLEEPING_TIME, places=2
            )
    
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')    
    def test_run_model_training_02(self,
                                   uuid_patch,
                                   repository_download_patch,
                                   model_manager_patch,
                                   builtin_exec_patch,
                                   builtin_eval_patch,
                                   repository_upload_patch,
                                   node_msg_patch):
        # test if all methods of `model` have been called
        TestRound.SLEEPING_TIME = 0
        URL_MSG = 'http://url/where/my/file?is=True'
        # initalisation of dummy classes
        class FakeUuid:
            # Fake uuid class
            def __init__(self):
                self.hex = 1234
                
            def __str__(self):
                return '1234'
            
        class FakeMonitor:
            def history(self) -> Dict[str, float]:
                return {'Loss': 0.1}
            
        class FakeNodeMessages:
            def __init__(self, msg: Dict[str, Any]):
                self.msg = msg

            def get_dict(self) -> Dict[str, Any]:
                return self.msg

        def repository_side_effect(model_url: str, model_name: str):
            return 200, 'my_python_model'
        
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeNodeMessages(msg)
            return fake_node_msg

        # initialisation of patchers 
        # dummy model
        class FakeModel:
            pass
        # Mocking model
        dummy_model = FakeModel
        #dummy_model.load = MagicMock()
        dummy_model.save = MagicMock()
        dummy_model.set_dataset = MagicMock()
        dummy_model.training_routine = MagicMock()
        dummy_model.after_training_params = MagicMock()
        #print(dummy_model.ll)
        
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        model_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = self.FakeModel
        repository_upload_patch.return_value = {'file': URL_MSG}
        node_msg_patch.side_effect = node_msg_side_effect
        
        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}
        dummy_monitor = MagicMock()
        self.r1.monitor = dummy_monitor        
        # action!
        with patch.object(self.FakeModel, 'training_routine') as mock1:
            msg = self.r1.run_model_training()
        # tests
        mock1.assert_called_once_with( monitor=dummy_monitor,
                                        node_args=None)
    
    def test_run_model_training_03(self):
        # test exceptions
        pass

    def test_run_model_training_04(self):
       # test case where model is not approved
       pass