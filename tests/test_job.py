import unittest
from unittest.mock import patch, MagicMock

import os

# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
import testsupport.mock_common_environ
# Import environ for node, since tests will be runing for node component
from fedbiomed.researcher.environ import environ

from fedbiomed.researcher.job import Job
from fedbiomed.researcher.responses import Responses

import torch
import numpy as np


class TestJob(unittest.TestCase):
    # once in test lifetime
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):

        self.patcher = patch('fedbiomed.researcher.requests.Requests.__init__',
                             return_value=None)
        self.patcher2 = patch('fedbiomed.common.repository.Repository.upload_file',
                              return_value={"file": environ['UPLOADS_URL']})
        self.patcher.start()
        self.patcher2.start()

    def tearDown(self) -> None:

        self.patcher.stop()
        self.patcher2.stop()
        #shutil.rmtree(os.path.join(VAR_DIR, "breakpoints"))
        # (above) remove files created during these unit tests


    # common mock classes
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


    # tests
    def test_job(self):
        '''

        '''
        # does not work yet !!
        #j = Job()

        pass


    def test_save_private_training_replies(self):
        """
        tests if `_save_training_replies` is properly extracting
        breakpoint info from `training_replies`. It uses a dummy class
        ResponsesMock, a weak implementation of `Responses` class
        """

        # mock model object
        model_file = MagicMock(return_value=None)
        model_file.save = MagicMock(return_value=None)

        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value={})

        # instantiate job
        test_job = Job(model=model_file,
                       data=fds)

        # first create a `_training_replies` variable
        training_replies = {
            0: self.ResponsesMock([]),
            1: self.ResponsesMock(
                [
                    {
                        "node_id": '1234',
                        'params': torch.Tensor([1, 3, 5]),
                        'dataset_id': 'id_node_1'
                    },
                    {
                        "node_id": '5678',
                        'params': np.array([1, 3, 5]),
                        'dataset_id': 'id_node_2'
                    },
                ])
            }

        # action
        new_training_replies = test_job._save_training_replies(training_replies)

        # check if `training_replies` is  saved accordingly
        self.assertTrue(type(new_training_replies) is list)
        self.assertTrue(len(new_training_replies) == 2)
        self.assertTrue('params' not in new_training_replies[1][0])
        self.assertEqual(new_training_replies[1][1].get('dataset_id'), 'id_node_2')


    @patch('fedbiomed.researcher.responses.Responses.__getitem__')
    @patch('fedbiomed.researcher.responses.Responses.__init__')
    def test_private_load_training_replies(
            self,
            patch_responses_init,
            patch_responses_getitem
            ):
        """tests if `_load_training_replies` is loading file content from path file.
        """

        # first test with a model done with pytorch
        pytorch_params = {
            # dont need other fields
            'model_params': torch.Tensor([1, 3, 5, 7])
        }
        sklearn_params = {
            # dont need other fields
            'model_params': np.array([[1,2,3,4,5], [2,8,7,5,5]])
        }
        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value={})

        # mock Pytorch model object
        model_torch = MagicMock(return_value=None)
        model_torch.save = MagicMock(return_value=None)
        func_torch_loadparams = MagicMock(return_value=pytorch_params)

        # instantiate job
        test_job_torch = Job(model=model_torch,
                             data=fds)
        # second create a `training_replies` variable
        loaded_training_replies =  [
                                        [
                                        {"success": True,
                                         "msg": "",
                                         "dataset_id": "dataset_1234",
                                         "node_id": "node_1234",
                                         "params_path": "/path/to/file/param.pt",
                                         "timing": {"time": 0}
                                         },
                                        {"success": True,
                                         "msg": "",
                                         "dataset_id": "dataset_4567",
                                         "node_id": "node_4567",
                                         "params_path": "/path/to/file/param2.pt",
                                         "timing": {"time": 0}
                                         }
                                        ]
                                    ]

        # mock Responses
        #
        # nota: works fine with one instance of Response active at a time
        # (which is not the case in `test_save_private_training_replies`)
        def side_responses_init(data):
            self.responses_data = data
        patch_responses_init.side_effect = side_responses_init
        patch_responses_init.return_value = None

        def side_responses_getitem(arg):
            return self.responses_data[arg]
        patch_responses_getitem.side_effect = side_responses_getitem


        # action
        torch_training_replies = test_job_torch._load_training_replies(
                                    loaded_training_replies,
                                    func_torch_loadparams
                                    )

        self.assertTrue(type(torch_training_replies) is dict)
        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            torch_training_replies[0][0]['node_id'],
            loaded_training_replies[0][0]['node_id'])
        # check `training_replies` for pytorch models
        self.assertTrue(torch.isclose(torch_training_replies[0][1]['params'],
                        pytorch_params['model_params']).all())
        self.assertTrue(torch_training_replies[0][1]['params_path'],
                        "/path/to/file/param2.pt")
        self.assertTrue(isinstance(torch_training_replies[0], Responses))


        ##### REPRODUCE TESTS BUT FOR SKLEARN MODELS
        # mock sklearn model object
        model_sklearn = MagicMock(return_value=None)
        model_sklearn.save = MagicMock(return_value=None)
        func_sklearn_loadparams = MagicMock(return_value=sklearn_params)
        # instantiate job
        test_job_sklearn = Job(model=model_sklearn,
                               data=fds)
        
        sklearn_training_replies = test_job_sklearn._load_training_replies(
                                        loaded_training_replies,
                                        func_sklearn_loadparams
                                        )

        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            sklearn_training_replies[0][0]['node_id'],
            loaded_training_replies[0][0]['node_id'])
        # check `training_replies` for sklearn models
        self.assertTrue(np.allclose(sklearn_training_replies[0][1]['params'],
                        sklearn_params['model_params']))
        self.assertTrue(sklearn_training_replies[0][1]['params_path'],
                        "/path/to/file/param2.pt")
        self.assertTrue(isinstance(sklearn_training_replies[0],
                                   Responses))

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
