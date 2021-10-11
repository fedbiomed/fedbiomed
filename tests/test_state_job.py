from typing import List
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

import os
import tempfile
import shutil

from fedbiomed.researcher.environ import UPLOADS_URL
from fedbiomed.researcher.job import Job

import torch
import numpy as np

def create_file(file_name: str):
    """create a file on the specified path `file_name`

    Args:
        file_name (str): path of the file
    """
    with open(file_name, "w") as f:
        f.write("this is a test- file. \
            This file should be removed at the end of unit tests")
   
class TestStateInJob(unittest.TestCase):
    def setUp(self):

        self.patcher = patch('fedbiomed.researcher.requests.Requests.__init__',
                             return_value=None)
        self.patcher2 = patch('fedbiomed.researcher.requests.Requests.search',
                              return_value=None)
        self.patcher3 = patch('fedbiomed.common.repository.Repository.upload_file',
                              return_value={"file": UPLOADS_URL})
        self.patcher.start() 
        self.patcher2.start()
        self.patcher3.start()

    def tearDown(self) -> None:
        
        self.patcher.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        #shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
        # (above) remove files created during these unit tests

    def test_save_private_training_replies(self):
        """
        tests if `_save_training_replies` is converted
        pytorch tensor and numpy arrays into path files (since
        both are not JSON serializable). It uses a dummy class
        ResponsesMock, a weak implementation of `Responses` class
        """
        class ResponsesMock(list):
            """
            This class mimicks `Responses` class. It 
            can be considered as a minimal implementation of 
            the forementioned class
            """
            def __init__(self, training_replies):
                super().__init__(training_replies)
                self._data = training_replies
            
            @property
            def data(self) -> list:
                """setter

                Returns:
                    list:  data of the class `Responses`
                """
                return(self._data)  
            def __getitem__(self, item):
                return self._data[item]
            
        model_file = MagicMock(return_value=None)
        
        model_file.save = MagicMock(return_value=None)
        test_job = Job(model=model_file)
        client_id1, client_id2 = '1', '2'
        tmpfilename_client1 = str(os.path.join('client', '1', 'path'))
        tmpfilename_client2 = str(os.path.join('client', '2', 'path'))
        
        test_job._params_path = {client_id1: tmpfilename_client1,
                                 client_id2: tmpfilename_client2}
                                 

        # first create a `_training_replies` variable
        _training_replies = {0: ResponsesMock([]),
                             1: ResponsesMock([
                                {"client_id": client_id1,
                                    'params': torch.Tensor([1, 3, 5]),
                                    'dataset_id': 'id_client_1'
                                },
                                {"client_id": client_id2,
                                    'params': np.array([1, 3, 5]),
                                    'dataset_id': 'id_client_2'
                                },
                                ])
                            }

        test_job._training_replies = _training_replies

        test_job._model_file = ""
        
        # second, use `save_state` function
        test_job.save_state(round=1)
        
        # FIXME: not the best way to check if path names are valid
        new_training_replies = test_job.state.get('training_replies')
        print(test_job.state)
 
        self.assertTrue(isinstance(new_training_replies[0].get('params'),
                                   str))
        self.assertEquals(new_training_replies[0].get('params'), tmpfilename_client1)
    

        

if __name__ == '__main__':  # pragma: no cover
   
    unittest.main()
