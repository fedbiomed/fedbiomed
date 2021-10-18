import unittest
from unittest.mock import patch, MagicMock

import os

from fedbiomed.researcher.environ import UPLOADS_URL
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.responses import Responses

import torch
import numpy as np
            

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
        tests if `_save_training_replies` is converting
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
        
        # mock model object
        model_file = MagicMock(return_value=None)
        model_file.save = MagicMock(return_value=None)
                
        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value=None)
        
        # instanciate job
        test_job = Job(model=model_file,
                       data=fds)
        # create dummy data mimicking 2 nodes
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
        
        # check if `training_replies` is  saved accordingly
        self.assertTrue(isinstance(new_training_replies[1][0].get('params'),
                                   str))
        self.assertEquals(new_training_replies[1][0].get('params'),
                          tmpfilename_client1)
    
        self.assertEquals(new_training_replies[1][1].get('dataset_id'),
                          'id_client_2')
    
    def test_private_load_training_replies(self):
        """tests if `_load_training_replies` is converting path files into
        pytorch tensor or numpy arrays.
        """
        
        # first test with a model done with pytorch
        pytorch_params = torch.Tensor([1, 3, 5, 7])
        sklearn_params = np.array([[1,2,3,4,5], [2,8,7,5,5]])
        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value=None)

        # mock Pytroch model object
        model_torch = MagicMock(return_value=None)
        model_torch.save = MagicMock(return_value=None)
        model_torch.load = MagicMock(return_value=pytorch_params)

        # instanciate job
        test_job_torch = Job(model=model_torch,
                             data=fds)
        # second create a `_training_replies` variable
        loaded_training_replies = {"3": [
                                        {"success": True,
                                         "msg": "",
                                         "dataset_id": "dataset_1234",
                                         "client_id": "client_1234",
                                         "params_path": "/path/to/file/param.pt",
                                         "params": "/path/to/file/param.pt",
                                         "timing": {"time": 0}
                                         },
                                        {"success": True,
                                         "msg": "",
                                         "dataset_id": "dataset_4567",
                                         "client_id": "client_4567",
                                         "params_path": "/path/to/file/param2.pt",
                                         "params": "/path/to/file/param2.pt",
                                         "timing": {"time": 0}
                                         }
                                        ]
                                   }
        
        params_path = {"client_1234": "/path/to/file/param.pt",
                       "client_4567": "/path/to/file/param2.pt"}
        test_job_torch._load_training_replies(loaded_training_replies,
                                              params_path)

        # check `self._training_replies` for pytorch models

        self.assertTrue(torch.isclose(test_job_torch._training_replies[3][0]['params'],
                        pytorch_params).all())
        self.assertTrue(test_job_torch._training_replies[3][0]['params_path'],
                        "/path/to/file/param.pt")
        self.assertTrue(isinstance(test_job_torch._training_replies[3],
                                   Responses))

        ##### REPRODUCE TESTS BUT FOR SKLEARN MODELS
        # mock sklearn model object
        model_sklearn = MagicMock(return_value=None)
        model_sklearn.save = MagicMock(return_value=None)
        model_sklearn.load = MagicMock(return_value=sklearn_params)
        # instanciate job
        test_job_sklearn = Job(model=model_sklearn,
                               data=fds)
        test_job_sklearn._load_training_replies(loaded_training_replies,
                                                params_path)

        # check `self._training_replies` for sklearn models
        self.assertTrue(np.allclose(test_job_sklearn._training_replies[3][0]['params'],
                        sklearn_params))
        self.assertTrue(test_job_sklearn._training_replies[3][0]['params_path'],
                        "/path/to/file/param.pt")
        self.assertTrue(isinstance(test_job_sklearn._training_replies[3],
                                   Responses))

if __name__ == '__main__':  # pragma: no cover
   
    unittest.main()
