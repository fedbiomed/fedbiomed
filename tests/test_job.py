import copy
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
        """tests if `_load_training_replies` is loading file content from path file
        and is building a proper training replies structure from breakpoint info
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

        # mock Responses
        #
        # nota: works fine only with one instance of Response active at a time thus
        # - cannot be used in `test_save_private_training_replies`
        # - if testing on more than 1 round, only the last round can be used for Asserts
        def side_responses_init(data, *args):
            self.responses_data = data
        def side_responses_getitem(arg, *args):
            return self.responses_data[arg]

        patch_responses_init.side_effect = side_responses_init
        patch_responses_init.return_value = None
        patch_responses_getitem.side_effect = side_responses_getitem


        # instantiate job
        test_job_torch = Job(model=model_torch,
                             data=fds)
        # second create a `training_replies` variable
        loaded_training_replies_torch =  [
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


        # action
        torch_training_replies = test_job_torch._load_training_replies(
                                    loaded_training_replies_torch,
                                    func_torch_loadparams
                                    )

        self.assertTrue(type(torch_training_replies) is dict)
        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            torch_training_replies[0][0]['node_id'],
            loaded_training_replies_torch[0][0]['node_id'])
        # check `training_replies` for pytorch models
        self.assertTrue(torch.isclose(torch_training_replies[0][1]['params'],
                        pytorch_params['model_params']).all())
        self.assertTrue(torch_training_replies[0][1]['params_path'],
                        "/path/to/file/param2.pt")
        self.assertTrue(isinstance(torch_training_replies[0], Responses))


        ##### REPRODUCE TESTS BUT FOR SKLEARN MODELS AND 2 ROUNDS
        
        # create a `training_replies` variable
        loaded_training_replies_sklearn =  [
                                        [
                                        {
                                            # dummy
                                            "params_path": "/path/to/file/param_sklearn.pt"
                                         }
                                        ], 
                                        [
                                        {"success": False,
                                         "msg": "",
                                         "dataset_id": "dataset_8888",
                                         "node_id": "node_8888",
                                         "params_path": "/path/to/file/param2_sklearn.pt",
                                         "timing": {"time": 6}
                                         }
                                        ]
                                    ]

        # mock sklearn model object
        model_sklearn = MagicMock(return_value=None)
        model_sklearn.save = MagicMock(return_value=None)
        func_sklearn_loadparams = MagicMock(return_value=sklearn_params)
        # instantiate job
        test_job_sklearn = Job(model=model_sklearn,
                               data=fds)
        

        # action
        sklearn_training_replies = test_job_sklearn._load_training_replies(
                                        loaded_training_replies_sklearn,
                                        func_sklearn_loadparams
                                        )

        # heuristic check `training_replies` for existing field in input
        self.assertEqual(
            sklearn_training_replies[1][0]['node_id'],
            loaded_training_replies_sklearn[1][0]['node_id'])
        # check `training_replies` for sklearn models
        self.assertTrue(np.allclose(sklearn_training_replies[1][0]['params'],
                        sklearn_params['model_params']))
        self.assertTrue(sklearn_training_replies[1][0]['params_path'],
                        "/path/to/file/param2_sklearn.pt")
        self.assertTrue(isinstance(sklearn_training_replies[0],
                                   Responses))


    @patch('fedbiomed.researcher.job.Job._load_training_replies')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    def test_load_state(
            self,
            patch_job_update_parameters,
            patch_job_load_training_replies
            ):
        """
        test if the job state values correctly initialize job
        """

        job_state = {
            'researcher_id': 'my_researcher_id_123456789',
            'job_id': 'my_job_id_abcdefghij',
            'model_params_path': '/path/to/my/model_file.py',
            'training_replies': { 0: 'un', 1: 'deux' }
        }
        new_training_replies = { 2: 'trois', 3: 'quatre' }

        # patch `update_parameters`
        patch_job_update_parameters.return_value = "dummy_string"

        # patch `_load_training_replies`
        patch_job_load_training_replies.return_value = new_training_replies

        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value={})

        # mock Pytorch model object
        model_object = MagicMock(return_value=None)
        model_object.save = MagicMock(return_value=None)

        # instantiate job
        test_job_torch = Job(model=model_object,
                             data=fds)


        
        # action
        test_job_torch.load_state(job_state)

        self.assertEqual(test_job_torch._researcher_id, job_state['researcher_id'])
        self.assertEqual(test_job_torch._id, job_state['job_id'])
        self.assertEqual(test_job_torch._training_replies, new_training_replies)



    @patch('fedbiomed.researcher.job.create_unique_link')
    @patch('fedbiomed.researcher.job.create_unique_file_link')
    @patch('fedbiomed.researcher.job.Job._save_training_replies')
    def test_save_state(
            self,
            patch_job_save_training_replies,
            patch_create_unique_file_link,
            patch_create_unique_link
            ):
        """
        test that job breakpoint state structure + file links are created
        """

        new_training_replies = [
            [
                { 'params_path': '/path/to/job_test_save_state_params0.pt' }
            ],
            [
                { 'params_path': '/path/to/job_test_save_state_params1.pt' }
            ]
        ]
        # expected transformed values of new_training_replies for save state
        new_training_replies_state = [
            [
                { 'params_path': 'xxx/job_test_save_state_params0.pt' }
            ],
            [
                { 'params_path': 'xxx/job_test_save_state_params1.pt' }
            ]
        ]
        
        link_path = '/path/to/job_test_save_state_params_link.pt'

        # patches configuration
        patch_create_unique_link.return_value = link_path
        
        def side_create_ufl(breakpoint_folder_path, file_path):
            return os.path.join(breakpoint_folder_path, os.path.basename(file_path))
        patch_create_unique_file_link.side_effect = side_create_ufl

        patch_job_save_training_replies.return_value = new_training_replies


        # mock FederatedDataSet
        fds = MagicMock()
        fds.data = MagicMock(return_value={})

        # mock Pytorch model object
        model_object = MagicMock(return_value=None)
        model_object.save = MagicMock(return_value=None)

        # instantiate job
        test_job_torch = Job(model=model_object,
                             data=fds)

        # choose arguments for saving state
        round = 3
        breakpoint_path = 'xxx'


        # action
        save_state = test_job_torch.save_state(breakpoint_path, round)

        self.assertEqual(environ['RESEARCHER_ID'], save_state['researcher_id'])
        self.assertEqual(test_job_torch._id, save_state['job_id'])
        self.assertEqual(link_path, save_state['model_params_path'])
        # check transformation of training replies
        for round_i, round in enumerate(new_training_replies):
            for response_i, _ in enumerate(round):
                self.assertEqual(
                    save_state['training_replies'][round_i][response_i]['params_path'], 
                    new_training_replies_state[round_i][response_i]['params_path'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
