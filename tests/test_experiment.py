# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.researcher.environ import environ

import unittest
from unittest.mock import patch
import os
import sys
import shutil
import json

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.experiment import Experiment

from tests.testsupport.fake_dataset import FederatedDataSetMock

class TestExperiment(unittest.TestCase):

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

        self.round_limit = 4
        self.tags = ['some_tag', 'more_tag']

        # useful for all tests, except load_breakpoint
        self.test_exp = Experiment(
            tags = self.tags,
            round_limit = self.round_limit,
            tensorboard=True,
            save_breakpoints=True)


    def tearDown(self) -> None:

        for patcher in self.patchers:
            patcher.stop()

        if environ['EXPERIMENTS_DIR'] in sys.path:
            sys.path.remove(environ['EXPERIMENTS_DIR'])

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
        round_current = 2
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
        self.test_exp._set_round_current(round_current)
        self.test_exp.breakpoint()
        

        # verification
        final_model_path = os.path.join(
            self.experimentation_folder_path, 
            'model_' + str("{:04d}".format(round_current)) + '.py')
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
        self.assertEqual(final_state['round_current'], round_current)
        self.assertEqual(final_state['round_limit'], self.round_limit)
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
            'round_current': round_current,
            'round_limit': self.round_limit,
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
        self.assertEqual(loaded_exp._round_current, round_current)
        self.assertEqual(loaded_exp._round_limit, self.round_limit)
        self.assertEqual(loaded_exp._experimentation_folder, self.experimentation_folder)
        self.assertEqual(loaded_exp._aggregator, final_aggregator)
        self.assertEqual(loaded_exp._node_selection_strategy, final_strategy)
        self.assertEqual(loaded_exp._tags, self.tags)
        self.assertEqual(loaded_exp._aggregated_params, final_aggregated_params)
        self.assertTrue(isinstance(loaded_exp._job, Job))
        self.assertEqual(self.job_state, final_job)


    def test_private_create_object(self):
        """tests `_create_object_ method : 
        Importing class, creating and initializing multiple objects from
        breakpoint state for object and file containing class code
        """
        
        # need EXPERIMENTS_DIR in PYTHONPATH to use it as directory for saving module
        sys.path.append(
            os.path.join(environ['EXPERIMENTS_DIR'],
            self.experimentation_folder)
            )

        # NO leading indents in class source code - write as string to avoid `re` operations
        class_source = \
            "class TestClass:\n" + \
            "   def __init__(self, **kwargs):\n" + \
            "       self._kwargs = kwargs\n" + \
            "   def load_state(self, state :str):\n" + \
            "       self._state = state\n"

        test_class_name = 'TestClass'
        module_name = 'testmodule'

        # arguments for creating object
        object_def = {
            'class': test_class_name,
            'module':  module_name,
            'other': 'my_arbitrary_field'
        }
        # optional object arguments
        object_kwargs = { 'toto': 321, 'titi': 'dummy_par' }

        # input file contains code for object creation
        module_file_path = os.path.join(self.experimentation_folder_path, module_name + '.py')
        with open(module_file_path, "w") as f:
            f.write(class_source)


        # action : instantiate multiple objects of the class
        loaded_objects_noarg = []
        for _ in range(0, 2):
            loaded_objects_noarg.append(Experiment._create_object(object_def))
        loaded_objects_arg = []
        for _ in range(0, 2):
            loaded_objects_arg.append(Experiment._create_object(object_def, **object_kwargs))

        # need to import class for testing
        exec('from ' + module_name + ' import ' + test_class_name)
        test_class = eval(test_class_name)

        # testing
        for loaded_object in loaded_objects_noarg:
            self.assertTrue(isinstance(loaded_object, test_class))
            self.assertEqual(loaded_object._state, object_def)
        for loaded_object in loaded_objects_arg:
            self.assertTrue(isinstance(loaded_object, test_class))
            self.assertEqual(loaded_object._state, object_def)
            self.assertEqual(loaded_object._kwargs, object_kwargs)

        # clean after tests
        del test_class


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
