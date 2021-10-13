import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import tempfile
import shutil
import json
from typing import Union
from fedbiomed.researcher.environ import TMP_DIR, VAR_DIR, UPLOADS_URL
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


def create_file(file_name: str):
    """creates a file on the specified path `file_name`

    Args:
        file_name (str): path of the file
    """
    with open(file_name, "w") as f:
        f.write("this is a test- file. \
            This file should be removed at the end of unit tests")

def load_json(file: str) -> Union[None, Exception]:
    """tests if a JSON file is parsable

    Args:
        file (str): path name of the json file to load
    Returns:
    
    """
    try:
        with open(file, "r") as f:
            json.load(f)
        return None
    except Exception as err:
        return err

# FIXME: it seems it is more an integration test than a unit test. 
# an improvement can be to 'patch' Job class instead of calling it. 
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
        self.patcher3 = patch('fedbiomed.common.repository.Repository.upload_file',
                              return_value={"file": UPLOADS_URL})
        self.patcher.start() 
        self.patcher2.start()
        self.patcher3.start()
        model_file = MagicMock(return_value=None)
        
        model_file.save_code = MagicMock(return_value=None)
        self.test_exp = Experiment(
            'some_tags', model_class=model_file, save_breakpoints=True
        )
        self.test_exp._create_breakpoints_folder()
        self.test_exp._create_breakpoint_exp_folder()
        
        self.test_exp._state_root_folder = VAR_DIR  # changing value of
        # the root folder

    def tearDown(self) -> None:
        
        self.patcher.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        try:
            shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
            # (above) remove files created during these unit tests
        except FileNotFoundError:
            pass
        

    def test_save(self):
        """tests:
        1. if model file is copied from temporary folder to breakpoint folder
        2. if state file created is json loadable
        """
        
        # Lets mock `client_selection_strategy`
        self.test_exp._client_selection_strategy = MagicMock(return_value=None)
        self.test_exp._client_selection_strategy.save_state = MagicMock(return_value={})
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmpdirname:
            def return_state(x=0):
                '''mimicking job.py 'save_state' method'''
                pass
            tempfile_path = os.path.join(tmpdirname, 'test_save')
            create_file(tempfile_path)
            self.test_exp._job.save_state = return_state
            self.test_exp._job.state = {"model_path": str(tempfile_path), 
                                        'params_path': {}}

            self.test_exp._save_state()
            exp_path = os.path.join(VAR_DIR,
                                    "breakpoints",
                                    "Experiment_0",
                                    'breakpoint_0')
            self.assertTrue(os.path.isdir(exp_path))  # test if folder used 
            # for saving the breakpoint exists
            test_model_path = os.path.join(exp_path, "breakpoint_0.json")
            
        # test if file containing state of experiment in breakpoint folder
        # is JSON loadable
        val = load_json(test_model_path)
        self.assertIs(val, None)
    
    def test_create_breakpoint(self,
                               breakpoint_folder_name: str="breakpoint_"):
        """
        Tests method `_create_breakpoint_file_and_folder`. Checks the correct
        spelling of breakpoint and state file.
        
        Args:
            breakpoint_folder_name (str, optional): [description]. Defaults to "breakpoint_".
        """
        
        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
                                                                    round=0)
        
        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(0))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(0) + ".json")
        
        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
                                                                    round=2)
        self.assertEqual(os.path.basename(bkpt_folder),
                         breakpoint_folder_name + str(2))
        self.assertEqual(bkpt_file,
                         breakpoint_folder_name + str(2) + ".json")
        

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
