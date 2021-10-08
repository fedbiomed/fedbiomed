import unittest
from unittest.mock import patch
import os
import tempfile
from fedbiomed.researcher.environ import TMP_DIR
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


class TestStateExp(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self) -> None:
        pass
    
    
    #@patch('fedbiomed.researcher.job.Job.__init__')
    @patch('fedbiomed.researcher.requests.Requests.search')
    def test_save_test(self, mocked_job_init):
        mocked_job_init.return_value = None
        
        test_exp = Experiment(
            'some_tags', save_breakpoints=True
        )
        test_exp._client_selection_strategy = DefaultStrategy(None)
        #test_exp._job.state = {}
        
        test_exp._create_breakpoints_folder()
        test_exp._create_breakpoint_exp_folder()
        with tempfile.TemporaryFile(dir=TMP_DIR) as temp_file:
            def return_state(x=0):
                pass
                #return {"model_path":temp_file}  # mimicing job 'save_state' method

            test_exp._job.save_state = return_state 
            test_exp._job.state = {"model_path":str(temp_file.name), 
                                   'params_path': {}}
            print(test_exp._job.save_state())
            test_exp._save_state()
        print("executed")
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()