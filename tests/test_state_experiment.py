import unittest
from unittest.mock import patch
import os
import tempfile
import shutil
from fedbiomed.researcher.environ import TMP_DIR, VAR_DIR
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


def create_file(file_name: str):
    """create a file on the specified path `file_name`

    Args:
        file_name (str): path of the file
    """
    with open(file_name, "w") as f:
        f.write("this is a test- file. \
            This file should be removed at the end of unit tests")
        

        
class TestStateExp(unittest.TestCase):
    def setUp(self):
        try:
            shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
            # clean up existing breakpoints
        except FileNotFoundError:
            pass
        self.patcher = patch('fedbiomed.researcher.requests.Requests.__init__',
                             return_value=None)
        self.patcher2 = patch('fedbiomed.researcher.requests.Requests.search',
                              return_value=None)
        self.patcher.start() 
        self.patcher2.start()

    def tearDown(self) -> None:
        
        self.patcher.stop()
        self.patcher2.stop()
        #shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
        # (above) remove files created during these unit tests

    def test_save(self):
        test_exp = Experiment(
            'some_tags', save_breakpoints=True
        )
        test_exp._client_selection_strategy = DefaultStrategy(None)
        
        test_exp._create_breakpoints_folder()
        test_exp._create_breakpoint_exp_folder()
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmpdirname:
            def return_state(x=0):
                '''mimicking job.py 'save_state' method'''
                pass
            tempfile_path = os.path.join(tmpdirname, 'test_save')
            create_file(tempfile_path)
            test_exp._job.save_state = return_state
            test_exp._job.state = {"model_path": str(tempfile_path), 
                                   'params_path': {}}

            test_exp._save_state()
            exp_path = os.path.join(VAR_DIR,
                                    "breakpoints",
                                    "Experiment_0",
                                    'breakpoint_0')
            self.assertTrue(os.path.isdir(exp_path))  # test if folder used 
            # for saving the breakpoint exists
            test_model_path = os.path.join(exp_path, "")
            

                
if __name__ == '__main__':  # pragma: no cover
    print(TMP_DIR)
    unittest.main()
