from copy import deepcopy
import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock, PropertyMock
import os
import tempfile
import shutil
import json
from typing import Union

# be sure to start from clean environment
from testsupport.delete_environ import delete_environ
delete_environ()
# use the test fake environ
import testsupport.mock_common_environ

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.experiment import Experiment


class TestStateExp(unittest.TestCase):

    def setUp(self):

        try:
            shutil.rmtree(environ['EXPERIMENTS_DIR'])
            # clean up existing experiments
        except FileNotFoundError:
            pass
        
        # folder name for experimentation in EXPERIMENT_DIR
        self.experimentation_folder = 'Experiment_101'
        self.experimentation_folder_path = \
            os.path.join(environ['EXPERIMENTS_DIR'], self.experimentation_folder)
        os.makedirs(self.experimentation_folder_path) 

        # build minimal objects, needed to extract state by calling object method
        class FederatedDataSetMock():
            def __init__(self, data):
                self._data = data
            def data(self):
                return self._data

        self.patchers = [
            patch('fedbiomed.researcher.requests.Requests.__init__',
                            return_value=None),
            patch('fedbiomed.researcher.requests.Requests.search',
                            return_value={}),
            # patch the whole class
            patch('fedbiomed.researcher.datasets.FederatedDataSet',
                            FederatedDataSetMock),
            patch('fedbiomed.researcher.experiment.create_exp_folder',
                            return_value=self.experimentation_folder),
            patch('fedbiomed.researcher.job.Job.__init__',
                            return_value=None),
            patch('fedbiomed.researcher.monitor.Monitor.__init__',
                            return_value=None),
            patch('fedbiomed.researcher.monitor.Monitor.on_message_handler',
                            return_value=False),
            patch('fedbiomed.researcher.requests.Requests.add_monitor_callback',
                            return_value=None)
        ]
        
        for patcher in self.patchers:
            patcher.start()

        self.rounds = 4
        self.tags = ['some_tag', 'more_tag']

        # useful for all tests, except load_breakpoint
        self.test_exp = Experiment(
            tags = self.tags,
            rounds = self.rounds,
            tensorboard=True,
            save_breakpoints=True)


    def tearDown(self) -> None:

        for patcher in self.patchers:
            patcher.stop()

        try:
            shutil.rmtree(environ['EXPERIMENTS_DIR'])
            # clean up existing experiments
        except FileNotFoundError:
            pass


    @patch('fedbiomed.researcher.experiment.create_unique_file_link')
    @patch('fedbiomed.researcher.experiment.create_unique_link')
    @patch('fedbiomed.researcher.job.Job.save_state')
    @patch('fedbiomed.researcher.job.Job.model_class')
    @patch('fedbiomed.researcher.job.Job.model_file')
    @patch('fedbiomed.researcher.experiment.choose_bkpt_file')
    # testing _save_breakpoint + _save_aggregated_params
    # (not exactly a unit test, but probably more interesting)
    def test_save_breakpoint(
            self,
            patch_choose_bkpt_file,
            patch_job_model_file,
            patch_job_model_class,
            patch_job_save_state,
            patch_create_ul,
            patch_create_ufl
            ):
        """tests `save_breakpoint` private method:
        1. if state file created is json loadable
        2. if state file content is correct
        """

        # name to for breakpoint file
        bkpt_file = 'my_breakpoint'
        # training data
        training_data = { 'node1': 'dataset1', 'node2': 'dataset2' }
        # we want to test with non null values
        training_args = { 'trarg1': 'my_string', 'trarg2': 444, 'trarg3': True }
        self.test_exp._training_args = training_args
        model_args = { 'modelarg1': 'value1', 'modelarg2': 234, 'modelarg3': False }
        self.test_exp._model_args = model_args
        model_file = '/path/to/my/model_file.py'
        model_class = 'MyOwnTrainingPlan'
        round_number = 2
        aggregator_state = { 'aggparam1': 'param_value', 'aggparam2': 987, 'aggparam3': True }
        strategy_state = { 'stratparam1': False, 'stratparam2': 'my_strategy', 'aggparam3': 0.45 }
        job_state = { 'jobparam1': { 'sub1': 1, 'sub2': 'two'}, 'jobparam2': 'myjob_value' }

        # aggregated_params
        agg_params = {
            'entry1': { 'params_path': '/dummy/path/to/aggparams/params_path.pt' },
            'entry2': { 'params_path': '/yet/another/path/other_params_path.pt' } 
        }
        self.test_exp._aggregated_params = agg_params

        # patch choose_bkpt_file create_unique_{file_}link  with minimal functions
        def side_bkpt_file(exp_folder, round):
            # save directly in experiment folder to avoir creating additional dirs
            return self.experimentation_folder_path, bkpt_file
        patch_choose_bkpt_file.side_effect = side_bkpt_file

        def side_create_ul(bkpt_folder_path, link_src_prefix, link_src_postfix, link_target_path):
            return os.path.join(bkpt_folder_path, link_src_prefix + link_src_postfix)
        patch_create_ul.side_effect = side_create_ul

        def side_create_ufl(bkpt_folder_path, file_path):
            return os.path.join(bkpt_folder_path, os.path.basename(file_path))
        patch_create_ufl.side_effect = side_create_ufl

        # patch Job state
        patch_job_save_state.return_value  = job_state

        # patch Job model_class / model_file
        self.test_exp._job.model_file = model_file
        self.test_exp._job.model_class = model_class

        # build minimal objects, needed to extract state by calling object method
        # (cannot just patch a method of a non existing object)
        class Aggregator():
            def save_state(self):
                return aggregator_state
        self.test_exp._aggregator = Aggregator()

        class Strategy():
            def save_state(self):
                return strategy_state
        self.test_exp._node_selection_strategy = Strategy()

        # use the mocked FederatedDataSet
        self.test_exp._fds = FederatedDataSet(training_data)

        # action
        self.test_exp._save_breakpoint(round_number)
        

        # verification
        final_model_path = os.path.join(
            self.experimentation_folder_path, 
            'model_' + str(round_number) + '.py')
        final_agg_params = {
            'entry1': {
                'params_path': os.path.join(self.experimentation_folder_path, 'params_path.pt')
                },
            'entry2': {
                'params_path': os.path.join(self.experimentation_folder_path, 'other_params_path.pt')
                } 
        }
        # better : catch exception if cannot read file or not json
        with open(os.path.join(self.experimentation_folder_path, bkpt_file), "r") as f:
            final_state = json.load(f)

        self.assertEqual(final_state['training_data'], training_data)
        self.assertEqual(final_state['training_args'], training_args)
        self.assertEqual(final_state['model_args'], model_args)
        self.assertEqual(final_state['model_path'], final_model_path)
        self.assertEqual(final_state['model_class'], model_class)
        self.assertEqual(final_state['round_number'], round_number + 1)
        self.assertEqual(final_state['round_number_due'], self.rounds)
        self.assertEqual(final_state['experimentation_folder'], self.experimentation_folder)
        self.assertEqual(final_state['aggregator'], aggregator_state)
        self.assertEqual(final_state['node_selection_strategy'], strategy_state)
        self.assertEqual(final_state['tags'], self.tags)
        self.assertEqual(final_state['aggregated_params'], final_agg_params)
        self.assertEqual(final_state['job'], job_state)


    @patch('fedbiomed.researcher.job.Job.load_state')
    @patch('fedbiomed.researcher.experiment.Experiment.model_instance.load')
    @patch('fedbiomed.researcher.experiment.Experiment.model_instance')
    @patch('fedbiomed.researcher.experiment.Experiment._create_object')
    @patch('fedbiomed.researcher.experiment.find_breakpoint_path')
    # test load_breakpoint + _load_aggregated_params + Experiment constructor
    # (not exactly a unit test, but probably more interesting)
    def test_load_breakpoint(self,
                             patch_find_breakpoint_path,
                             patch_create_object,
                             patch_model_instance,
                             patch_model_instance_load,
                             patch_job_load_state
                             ):
        """ test `load_breakpoint` :
            1. if breakpoint file is json loadable
            2. if experiment is correctly configured from breakpoint
        """

        # breakpoint values
        bkpt_file = 'file_4_breakpoint'

        training_data = { 'train_node1': 'my_first_dataset', 2: 243 }
        training_args = { 1: 'my_first arg', 'training_arg2': 123.45 }
        model_args = { 'modarg1': True, 'modarg2': 7.12, 'modarg3': 'model_param_foo' }
        model_path = '/path/to/breakpoint_model_file.py'
        model_class = 'ThisIsTheTrainingPlan'
        round_current = 1
        experimentation_folder = 'My_experiment_folder_258'
        aggregator = { 'aggreg1': False, 'aggreg2': 'dummy_agg_param', 18: 'agg_param18' }
        strategy = { 'strat1': 'test_strat_param', 'strat2': 421, 3: 'strat_param3' }
        aggregated_params = {
            '1': { 'params_path': '/path/to/my/params_path_1.pt' },
            2: { 'params_path': '/path/to/my/params_path_2.pt' }
        }
        job = { 1: 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999 }

        # breakpoint structure
        state = {
            'training_data': training_data,
            'training_args': training_args,
            'model_args': model_args,
            'model_path': model_path,
            'model_class': model_class,
            'round_number': round_current + 1,
            'round_number_due': self.rounds,
            'experimentation_folder': experimentation_folder,
            'aggregator': aggregator,
            'node_selection_strategy': strategy,
            'tags': self.tags,
            'aggregated_params': aggregated_params,
            'job': job
        }
        # create breakpoint file
        with open(os.path.join(self.experimentation_folder_path, bkpt_file), "w") as f:
            json.dump(state, f)

        # mocked model params
        model_params = { 'something.bias': [12, 14], 'something.weight': [13, 15] }


        # target aggregated params
        final_aggregated_params = {
            1: { 'params_path': '/path/to/my/params_path_1.pt' },
            2: { 'params_path': '/path/to/my/params_path_2.pt' }
        }
        for aggpar in final_aggregated_params.values():
            aggpar['params'] = model_params
        # target breakpoint element arguments
        final_training_data = { 'train_node1': 'my_first_dataset', '2': 243 }
        final_training_args = { '1': 'my_first arg', 'training_arg2': 123.45 }
        final_aggregator = { 'aggreg1': False, 'aggreg2': 'dummy_agg_param', '18': 'agg_param18' }
        final_strategy = { 'strat1': 'test_strat_param', 'strat2': 421, '3': 'strat_param3' }
        final_job = { '1': 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999 }


        # patch functions for loading breakpoint
        patch_find_breakpoint_path.return_value = self.experimentation_folder_path, bkpt_file
        def side_create_object(args, **kwargs):
            return args
        patch_create_object.side_effect = side_create_object
        patch_model_instance_load.return_value = model_params

        # keep job state to ensure it was properly initialized
        self.job_state = None
        def side_job_load_state(job_state):
            self.job_state = job_state
        patch_job_load_state.side_effect = side_job_load_state

        
        # action
        loaded_exp = Experiment.load_breakpoint(self.experimentation_folder_path)

        # verification
        self.assertTrue(isinstance(loaded_exp, Experiment))
        self.assertTrue(isinstance(loaded_exp._fds, FederatedDataSet))
        self.assertEqual(loaded_exp._fds.data(), final_training_data)
        self.assertEqual(loaded_exp._training_args, final_training_args)
        self.assertEqual(loaded_exp._model_args, model_args)
        self.assertEqual(loaded_exp._model_path, model_path)
        self.assertEqual(loaded_exp._model_class, model_class)
        self.assertEqual(loaded_exp._round_init, round_current + 1)
        self.assertEqual(loaded_exp._rounds, self.rounds)
        self.assertEqual(loaded_exp._experimentation_folder, self.experimentation_folder)
        self.assertEqual(loaded_exp._aggregator, final_aggregator)
        self.assertEqual(loaded_exp._node_selection_strategy, final_strategy)
        self.assertEqual(loaded_exp._tags, self.tags)
        self.assertEqual(loaded_exp._aggregated_params, final_aggregated_params)
        self.assertTrue(isinstance(loaded_exp._job, Job))
        self.assertEqual(self.job_state, final_job)



        ## tests
        #patch_json_load.assert_called_once()  # check if patched
        ## json has been called
        #self.assertTrue(isinstance(loaded_exp, Experiment))
        #self.assertEqual(loaded_exp._round_init,
        #                 loaded_states.get('round_number'))
        #self.assertEqual(loaded_exp._job._id,
        #                 loaded_states.get('job_id'))
        #self.assertEqual(loaded_exp._rounds,
        #                 loaded_states.get('round_number_due'))


#    def test_create_breakpoint(self,
#                               breakpoint_folder_name: str="breakpoint_"):
#        """
#        Tests method `_create_breakpoint_file_and_folder`. Checks the correct
#        spelling of breakpoint and state file.
#
#        Args:
#            breakpoint_folder_name (str, optional): [description]. Defaults to "breakpoint_".
#        """
#
#        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
#                                                                    round=0)
#
#        self.assertEqual(os.path.basename(bkpt_folder),
#                         breakpoint_folder_name + str(0))
#        self.assertEqual(bkpt_file,
#                         breakpoint_folder_name + str(0) + ".json")
#
#        bkpt_folder, bkpt_file = self.test_exp._create_breakpoint_file_and_folder(
#                                                                    round=2)
#        self.assertEqual(os.path.basename(bkpt_folder),
#                         breakpoint_folder_name + str(2))
#        self.assertEqual(bkpt_file,
#                         breakpoint_folder_name + str(2) + ".json")
#
#    def test_private_get_latest_file(self):
#        """tests if `_get_latest_file` returns more recent
#        file"""
#
#        # test 1
#        files = ["Experiment_0",
#                 "Experiment_4",
#                 "EXperiment_5",
#                 "blabla",
#                 "99_blabla"]
#
#        pathfile_test = "/path/to/a/file"
#
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=False)
#        self.assertEqual(files[2], latest_file)
#
#        # test 2: in this test, we patch isir builtin function
#        # so it returns always `True` when called
#        patcher_builtin_os_path_isdir = patch("os.path.isdir",
#                                              return_value=True)
#        patcher_builtin_os_path_isdir.start()
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=True)
#        self.assertEqual(files[2], latest_file)
#        patcher_builtin_os_path_isdir.stop()
#        # test 3
#        files = []
#        latest_file = Experiment._get_latest_file(pathfile_test,
#                                                  files,
#                                                  only_folder=False)
#        self.assertEqual(latest_file, None)
#
#        # test 4: test if exception is raised
#        files = ['q', 'foo', 'bar']
#
#        self.assertRaises(FileNotFoundError,
#                          Experiment._get_latest_file,
#                          pathfile_test,
#                          files,
#                          only_folder=False)
#
#    def test_private_import_module(self):
#        args = {"class": "myclass",
#                "module": "module.containing.my.class",
#                "parameters": None}
#
#        # test 1: default
#        import_str = Experiment._import_module(args)
#        self.assertEqual(import_str,
#                         'from module.containing.my.class import myclass')
#        # test 2: custom module
#        args = {"class": "myclass",
#                "module": "custom",
#                "parameters": None}
#        import_str = Experiment._import_module(args)
#        self.assertEqual(import_str, 'import myclass')
#
#    @patch('fedbiomed.researcher.job.Job._load_training_replies')
#    def test_private_load_training_replies(self,
#                                           path_job_load_training_replies):
#        path_job_load_training_replies.return_value = None
#        tr_replies = {1: [{"foo": "bar"}]}
#        pr_path = ["/path/to/file", "/path/to/file"]
#        self.test_exp._load_training_replies(tr_replies, pr_path)
#        # test if Job's `_load_training_replies` has been called once
#        path_job_load_training_replies.assert_called_once()
#
#    @patch('fedbiomed.researcher.job.Job._load_training_replies')
#    @patch('fedbiomed.researcher.job.Job.__init__')
#    @patch('fedbiomed.researcher.experiment.eval')
#    @patch('fedbiomed.researcher.experiment.Experiment._import_module')
#    @patch('json.load')
#    @patch("builtins.open")
#    @patch('fedbiomed.researcher.experiment.Experiment._find_breakpoint_path')
#    def test_load_breakpoint(self,
#                             patch_find_breakpoint_path,
#                             patch_builtin_open,
#                             patch_json_load,
#                             patch_import_module,
#                             patch_builtin_eval,
#                             patch_job_init,
#                             patch_job_load_training_replies
#                             ):
#
#        values = ["/path/to/breakpoint/folder", "my_breakpoint.json"]
#        dummy_agg = {"class":None,
#                     "Module":None}
#        loaded_states = {
#            "node_selection_strategy": dummy_agg,
#            "aggregator": dummy_agg,
#            "tags": ["some_tags"],
#            "node_id": "m_node_id",
#            "model_class": "my_model_class",
#            "model_path": "/path/to/model/file",
#            "model_args": {},
#            "training_args":{},
#            "round_number": 1,
#            "round_number_due":3,
#            "training_data": {},
#            "job_id": "1234",
#            "researcher_id": '1234',
#            "params_path": [],
#            "training_replies": {"1":[{}, {}]}
#
#        }
#        patch_find_breakpoint_path.return_value = values
#        patch_builtin_open.return_value = MagicMock()
#        patch_json_load.return_value = loaded_states
#        patch_import_module.return_value = "import abc"  # not sure it is a good idea
#        patch_builtin_eval.return_value = MagicMock()
#        patch_job_init.return_value = None
#        patch_job_load_training_replies.return_value = MagicMock()
#
#        bkpt_folder = "/path/to/breakpoint/folder"
#        loaded_exp = Experiment.load_breakpoint(bkpt_folder)
#        print(type(loaded_exp))
#
#        # tests
#        patch_json_load.assert_called_once()  # check if patched
#        # json has been called
#        self.assertTrue(isinstance(loaded_exp, Experiment))
#        self.assertEqual(loaded_exp._round_init,
#                         loaded_states.get('round_number'))
#        self.assertEqual(loaded_exp._job._id,
#                         loaded_states.get('job_id'))
#        self.assertEqual(loaded_exp._rounds,
#                         loaded_states.get('round_number_due'))
#
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_1(self,
#                                            patch_os_listdir,
#                                            patch_os_path_isdir
#                                            ):
#        # test 1 : test if results are corrects  if path
#        # to breakpoint has been given by user
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#
#        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(bkpt_folder)
#        self.assertEqual(bkpt_folder, bkpt_folder_out)
#        self.assertEqual(state_file, 'breakpoint_1234.json')
#
#    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_2(self,
#                                            patch_os_listdir,
#                                            patch_os_path_isdir,
#                                            patch_get_latest_file
#                                            ):
#        # test 2 : test if path to breakpoint has not been given by user
#        # ie set to None
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#        patch_get_latest_file.return_value = "breakpoint"
#        latest_bkpt_folder = os.path.join(environ['EXPERIMENTS_DIR'],
#                                          'breakpoint',
#                                          'breakpoint')
#
#        bkpt_folder_out, state_file = Experiment._find_breakpoint_path(None)
#        self.assertEqual(state_file, 'breakpoint_1234.json')
#        self.assertEqual(bkpt_folder_out, latest_bkpt_folder)
#
#    @patch('os.path.isfile')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_raise_err1(self,
#                                                     patch_os_listdir,
#                                                     patch_os_path_isdir,
#                                                     patch_os_path_isfile):
#        # triggers error: FileNotFoundError, error is not a folder
#        # but a file
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = False
#        patch_os_path_isfile.return_value = False
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#    @patch('os.path.isfile')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_path_raise_err2(self,
#                                                     patch_os_listdir,
#                                                     patch_os_path_isdir,
#                                                     patch_os_path_isfile):
#        # triggers error: FileNotFoundError (folder not found)
#        #
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['breakpoint_1234.json',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = False
#        patch_os_path_isfile.return_value = True
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#
#    @patch('fedbiomed.researcher.experiment.Experiment._get_latest_file')
#    @patch('os.path.isdir')
#    @patch('os.listdir')
#    def test_private_find_breakpoint_raise_err_3(self,
#                                                 patch_os_listdir,
#                                                 patch_os_path_isdir,
#                                                 patch_get_latest_file
#                                                 ):
#        # test 3 : test if rerror is raised when json file
#        # not found in a breakpoint folder specified by user
#        bkpt_folder = "/path/to/breakpoint"
#        patch_os_listdir.return_value = ['one_file',
#                                         "another_file"]
#        patch_os_path_isdir.return_value = True
#        patch_get_latest_file.return_value = "breakpoint"
#
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          bkpt_folder)
#
#    def test_private_find_breakpoint_raise_err_4(self):
#        # test 4 : test if rerror is raised when latest
#        # file has not been foud
#        self.assertRaises(FileNotFoundError,
#                          Experiment._find_breakpoint_path,
#                          None)
#
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
