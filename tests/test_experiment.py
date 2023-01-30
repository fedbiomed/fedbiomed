from typing import Dict
import unittest
import os
import sys
import shutil
import json
import inspect

from unittest.mock import patch, MagicMock, PropertyMock

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################


from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.secagg import SecaggBiprimeContext, SecaggContext, SecaggServkeyContext
import testsupport.fake_researcher_environ  ## noqa (remove flake8 false warning)

from testsupport.fake_dataset import FederatedDataSetMock
from testsupport.fake_experiment import ExperimentMock
from testsupport.fake_training_plan import FakeModel
from testsupport.base_fake_training_plan import BaseFakeTrainingPlan
from testsupport.fake_researcher_secagg import FAKE_CONTEXT_VALUE, FakeSecaggServkeyContext, \
    FakeSecaggBiprimeContext

from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.exceptions import FedbiomedSilentTerminationError

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.scaffold import Scaffold
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
import fedbiomed.researcher.experiment
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


class FakeAggregator(Aggregator):
    aggregator_name: str = 'dummy-aggregator'

class FakeStrategy(Strategy):
    pass
        

class TestExperiment(ResearcherTestCase):
    """ Test for Experiment class """

    # For testing training_plan setter of Experiment
    class FakeModelTorch(BaseFakeTrainingPlan):
        """ Should inherit TorchTrainingPlan to pass the condition
            `issubclass` of `TorchTrainingPlan`
        """
        def init_model(self, args):
            pass

        pass

    @staticmethod
    def create_fake_training_plan_file(name: str):
        """ Class method saving codes of FakeModel

        Args:
            name (str): Name of the model file that will be created
        """

        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        tmp_dir_model = os.path.join(tmp_dir, name)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        content = "from typing import Dict, Any, List\n"
        content += "import time\n"
        content += inspect.getsource(FakeModel)
        file = open(tmp_dir_model, "w")
        file.write(content)
        file.close()

        return tmp_dir_model


    @classmethod
    def setUpClass(cls) -> None:

        super().setUpClass()

        # Create FakeAggregator that does not have subclass
        class FakeAggregator:
            pass

        # Create FakeStrategy does not have subclass
        class FakeStrategy:
            pass

        cls.fake_strategy = FakeStrategy
        cls.fake_aggregator = FakeAggregator


    def setUp(self):

        try:
            # clean up existing experiments
            shutil.rmtree(environ['EXPERIMENTS_DIR'])

        except FileNotFoundError:
            pass

        # folder name for experimentation in EXPERIMENT_DIR
        self.experimentation_folder = 'Experiment_101'
        self.experimentation_folder_path = \
            os.path.join(environ['EXPERIMENTS_DIR'], self.experimentation_folder)
        os.makedirs(self.experimentation_folder_path)

        # Define patchers
        # Patchers that are not required be modified during the tests
        self.patchers = [
            patch('fedbiomed.researcher.datasets.FederatedDataSet',
                  FederatedDataSetMock),
            patch('fedbiomed.researcher.aggregators.aggregator.Aggregator.__init__',
                  return_value=None),
        ]

        self.monitor_mock = MagicMock(return_value=None)
        self.monitor_mock.on_message_handler = MagicMock()
        self.monitor_mock.close_writer = MagicMock()

        # Patchers that required be modified during the tests
        self.patcher_monitor_init = patch('fedbiomed.researcher.monitor.Monitor', MagicMock(return_value=None))
        self.patcher_monitor_on_message_handler = patch('fedbiomed.researcher.monitor.Monitor.on_message_handler',
                                                        MagicMock(return_value=None))
        self.patcher_monitor_close_writer = patch('fedbiomed.researcher.monitor.Monitor.close_writer',
                                                  MagicMock(return_value=None))
        self.patcher_cr_folder = patch('fedbiomed.researcher.experiment.create_exp_folder',
                                       return_value=self.experimentation_folder)
        self.patcher_job = patch('fedbiomed.researcher.job.Job.__init__', MagicMock(return_value=None))
        self.patcher_logger_info = patch('fedbiomed.common.logger.logger.info', MagicMock(return_value=None))
        self.patcher_logger_error = patch('fedbiomed.common.logger.logger.error', MagicMock(return_value=None))
        self.patcher_logger_critical = patch('fedbiomed.common.logger.logger.critical', MagicMock(return_value=None))
        self.patcher_logger_debug = patch('fedbiomed.common.logger.logger.debug', MagicMock(return_value=None))
        self.patcher_logger_warning = patch('fedbiomed.common.logger.logger.warning', MagicMock(return_value=None))
        self.patcher_request_init = patch('fedbiomed.researcher.requests.Requests.__init__',
                                          MagicMock(return_value=None))
        self.patcher_request_search = patch('fedbiomed.researcher.requests.Requests.search', MagicMock(return_value={}))



        for patcher in self.patchers:
            patcher.start()

        # Define mocks from patchers

        self.mock_monitor_init = self.patcher_monitor_init.start()
        self.mock_monitor_on_message = self.patcher_monitor_on_message_handler.start()
        self.mock_monitor_close_writer = self.patcher_monitor_close_writer.start()
        self.mock_create_folder = self.patcher_cr_folder.start()
        self.mock_logger_info = self.patcher_logger_info.start()
        self.mock_logger_error = self.patcher_logger_error.start()
        self.mock_logger_critical = self.patcher_logger_critical.start()
        self.mock_logger_debug = self.patcher_logger_debug.start()
        self.mock_logger_warning = self.patcher_logger_warning.start()
        self.mock_job = self.patcher_job.start()
        self.mock_request_init = self.patcher_request_init.start()
        self.mock_request_search = self.patcher_request_search.start()

        self.round_limit = 4
        self.tags = ['some_tag', 'more_tag']
        self.nodes = ['node-1', 'node-2']

        # useful for all tests, except load_breakpoint
        self.test_exp = Experiment(
            nodes=['node-1', 'node-2'],
            tags=self.tags,
            round_limit=self.round_limit,
            tensorboard=True,
            save_breakpoints=True)

    def tearDown(self) -> None:

        # Stop patchers patched in array
        for patcher in self.patchers:
            patcher.stop()

        # Stop patchers
        self.patcher_cr_folder.stop()
        self.patcher_job.stop()
        self.patcher_logger_info.stop()
        self.patcher_logger_error.stop()
        self.patcher_logger_critical.stop()
        self.patcher_request_init.stop()
        self.patcher_request_search.stop()
        self.patcher_logger_debug.stop()
        self.patcher_logger_warning.stop()
        self.patcher_monitor_init.stop()
        self.patcher_monitor_on_message_handler.stop()
        self.patcher_monitor_close_writer.stop()

        if environ['EXPERIMENTS_DIR'] in sys.path:
            sys.path.remove(environ['EXPERIMENTS_DIR'])

        try:
            shutil.rmtree(environ['EXPERIMENTS_DIR'])
            # clean up existing experiments
        except FileNotFoundError:
            pass

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


    def assertSubDictInDict(self, sub_dict: Dict, dict: Dict, msg: str = ''):
        ok_array = [False] * len(sub_dict)
        for i, (s_key, s_val) in enumerate(sub_dict.items()):
            for key, val in dict.items():
                if s_key == key and val == s_val:
                    ok_array[i] = True
        assert all(ok_array), msg + f'{sub_dict} is not in {dict}'

    def test_experiment_01_getters(self):
        """ Testings getters of Experiment class """

        # Tags should be ['some_tag', 'more_tag'] that comes from setUp
        tags = self.test_exp.tags()
        self.assertListEqual(tags, self.tags, 'Can not get tags from Experiment properly')

        # Test getter for nodes
        nodes = self.test_exp.nodes()
        self.assertListEqual(nodes, self.nodes, 'Can not get nodes from Experiment properly')

        # Test getter for training_data
        fds = self.test_exp.training_data()
        self.assertIsInstance(fds, FederatedDataSet, 'The getter `.training_data` does not return '
                                                     'proper FederatedDataset instance')

        # Test getter for aggregator
        aggregator = self.test_exp.aggregator()
        self.assertIsInstance(aggregator, FedAverage, 'The getter `.aggregator()` does not return proper '
                                                      'FedAverage instance')
        self.assertIsInstance(aggregator, Aggregator, 'The getter `.aggregator()` does not return proper '
                                                      'Aggregator instance')

        # Test getter for strategy
        strategy = self.test_exp.strategy()
        self.assertIsInstance(strategy, Strategy, 'The getter `.strategy()` does not return proper '
                                                  'Strategy instance')
        self.assertIsInstance(strategy, DefaultStrategy, 'The getter `.strategy()` does not return proper '
                                                         'DefaultStrategy instance')

        # Test getter round_limit
        round_limit = self.test_exp.round_limit()
        self.assertEqual(round_limit, self.round_limit, 'Getter for round_limit did not return correct round'
                                                        'limit.')

        # Test getter for round_current
        round_current = self.test_exp.round_current()
        self.assertEqual(round_current, 0, 'Getter for round_current did not return correct round')

        # Test getter for experimentation_folder
        exp_folder = self.test_exp.experimentation_folder()
        self.assertEqual(exp_folder, 'Experiment_101', 'Getter for experimentation folder did not return correct '
                                                       'folder name')

        # Test getter for experimentation folder
        expected_exp_path = os.path.join(environ['EXPERIMENTS_DIR'], exp_folder)
        exp_path = self.test_exp.experimentation_path()
        self.assertEqual(exp_path, expected_exp_path, 'Getter for experimentation folder did not return expected '
                                                      'experimentation path')

        # Test getter for training_plan
        training_plan = self.test_exp.training_plan_class()
        self.assertIsNone(training_plan, 'Getter for training_plan did not return expected training_plan')

        # Test getter for training_plan_path
        training_plan_path = self.test_exp.training_plan_path()
        self.assertIsNone(training_plan_path, 'Getter for training_plan_path did not return expected training_plan_path')

        # Test getter for model arguments
        model_args = self.test_exp.model_args()
        self.assertDictEqual(model_args, {}, 'Getter for model_args did not return expected value')

        # Test getter fpr Job instance
        job = self.test_exp.job()
        self.assertIsNone(job, 'Getter for Job did not return expected')

        # Test getter for save_breakpoints
        save_breakpoints = self.test_exp.save_breakpoints()
        self.assertTrue(save_breakpoints, 'Getter for save_breakpoints did not return expected value: True')

        # Test getter for monitor
        monitor = self.test_exp.monitor()
        self.assertIsInstance(monitor, Monitor, 'Getter for monitor did not return expected Monitor instance')

        # Test getter for aggregated params
        agg_params = self.test_exp.aggregated_params()
        self.assertDictEqual(agg_params, {}, 'Getter for aggregated_params did not return expected value: {}')

        # Test getter training_replies

        # Test when ._job is None
        training_replies = self.test_exp.training_replies()
        self.assertIsNone(training_replies, 'Getter for training_replies did not return expected value None')
        self.mock_logger_error.assert_called_once()

        # Test when ._job is not None
        tr_reply = {'node-1': True}
        type(self.mock_job).training_replies = PropertyMock(return_value=tr_reply)
        self.test_exp._job = self.mock_job
        training_replies = self.test_exp.training_replies()
        self.assertDictEqual(training_replies, tr_reply, 'Getter for training_replies did not return expected values')

        # Test getter for model instance

        # Test when ._job is None
        self.test_exp._job = None
        self.mock_logger_error.reset_mock()
        training_plan = self.test_exp.training_plan()
        self.assertIsNone(training_plan, 'Getter for .training_plan did not return expected value None')
        self.mock_logger_error.assert_called_once()

        # Test when ._job is not None
        fake_training_plan = FakeModel()
        type(self.mock_job).training_plan = PropertyMock(return_value=fake_training_plan)
        self.test_exp._job = self.mock_job
        training_plan = self.test_exp.training_plan()
        self.assertEqual(training_plan, fake_training_plan, 'Getter for training_plan did not return expected Model')

        # Test getters for testing arguments
        test_ratio = self.test_exp.test_ratio()
        self.assertEqual(test_ratio, 0, 'Getter for test ratio has returned unexpected value')

        test_metric = self.test_exp.test_metric()
        # Should be None since there is no metric has been set
        self.assertIsNone(test_metric, 'Getter for test metric has returned unexpected value')

        test_metric_arg = self.test_exp.test_metric_args()
        # Should be empty Dict since there is no metric has been set
        self.assertDictEqual(test_metric_arg, {}, 'Getter for test metric args has returned unexpected value')

        test = self.test_exp.test_on_global_updates()
        # Should be false
        self.assertEqual(test, False, 'Getter for test on global update args has returned unexpected value')

        test = self.test_exp.test_on_local_updates()
        # Should be false
        self.assertEqual(test, False, 'Getter for test on local updates has returned unexpected value')

        test = self.test_exp.use_secagg()
        # Should be false
        self.assertEqual(test, False, 'Getter for secagg usage has returned unexpected value')

        test1, test2 = self.test_exp.secagg_context()
        self.assertEqual(test1, None, 'Getter for test on secagg context has returned unexpected value')
        self.assertEqual(test2, None, 'Getter for test on secagg context has returned unexpected value')

    def test_experiment_02_info(self):
        """Testing the method .info() of experiment class """
        self.test_exp.info()

        # Test info by completing missing parts for proper .run
        self.test_exp._fds = FederatedDataSetMock({'node-1': []})
        self.test_exp._job = self.mock_job
        self.test_exp._training_plan_is_defined = True
        self.test_exp.info()


    @patch('builtins.eval')
    @patch('builtins.print')
    def test_experiment_02_info_exception(self, mock_print, mock_eval):
        """Testing exceptions raise in info method of Experiment """

        mock_print.return_value(None)
        mock_eval.side_effect = Exception
        with self.assertRaises(SystemExit):
            self.test_exp.info()


    def test_experiment_03_set_tags(self):
        """ Testing setter for _tags attribute of Experiment """

        # Test tags as List
        tags_expected = ['tags-1', 'tag-2']
        tags = self.test_exp.set_tags(tags_expected)
        self.assertListEqual(tags, tags_expected, 'Setter for tags can not set tags properly')

        # Test tags as String
        tags_expected = 'tag-1'
        tags = self.test_exp.set_tags(tags_expected)
        self.assertListEqual(tags, [tags_expected], 'Setter for tags can not set tags properly when tags argument is '
                                                    'in string type')

        # Test bad type of tags
        tags_expected = MagicMock(return_value=None)
        with self.assertRaises(SystemExit):
            tags = self.test_exp.set_tags(tags_expected)

        # Test bad types in tags array
        tags_expected = [{}, {}]
        with self.assertRaises(SystemExit):
            tags = self.test_exp.set_tags(tags_expected)

        # Test set tags as none
        tags_expected = None
        tags = self.test_exp.set_tags(tags_expected)
        self.assertEqual(tags, tags_expected, f'Expected tags should be None not {tags}')


    def test_experiment_04_set_nodes(self):

        # Test tags as List
        nodes_expected = ['node-1', 'node-2']
        nodes = self.test_exp.set_nodes(nodes_expected)
        self.assertListEqual(nodes, nodes_expected, 'Setter for nodes can not set nodes properly')

        # Test bad type of nodes
        nodes_expected = MagicMock(return_value=None)
        with self.assertRaises(SystemExit):
            nodes = self.test_exp.set_nodes(nodes_expected)

        # Test nodes as String (bad type)
        nodes_expected = 'tag-1'
        with self.assertRaises(SystemExit):
            nodes = self.test_exp.set_nodes(nodes_expected)

        # Test bad types in nodes array
        nodes_expected = [{}, {}]
        with self.assertRaises(SystemExit):
            nodes = self.test_exp.set_nodes(nodes_expected)

        # Test raising SilentTerminationError
        with patch.object(fedbiomed.researcher.experiment, 'is_ipython',
                          create=True) as m:
            m.return_value = True

            with self.assertRaises(FedbiomedSilentTerminationError):
                self.test_exp.set_nodes(nodes_expected)

            nodes_expected = 'tag-1'
            with self.assertRaises(FedbiomedSilentTerminationError):
                nodes = self.test_exp.set_nodes(nodes_expected)

        # Test set nodes as none
        nodes_expected = None
        nodes = self.test_exp.set_nodes(nodes_expected)
        self.assertEqual(nodes, nodes_expected, f'Expected nodes should be None not {nodes}')


    def test_experiment_04_set_training_data(self):
        """ Testing setter for ._fds attribute of Experiment """
        # Test by passing training data as None when there are tags already set
        td_expected = None
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertIsNone(training_data, 'Setter for training_data did not set training data to None')

        # Test by passing training data as None and from_tags `True`
        td_expected = None
        training_data = self.test_exp.set_training_data(training_data=td_expected, from_tags=True)
        self.assertIsInstance(training_data, FederatedDataSet, 'Setter for training_data is not set properly')

        # Test by passing training data as Node when the tags is None
        td_expected = None
        _ = self.test_exp.set_tags(tags=None)
        training_data = self.test_exp.set_training_data(training_data=td_expected, from_tags=True)
        self.assertEqual(training_data, td_expected, 'Setter for training data is not set as expected: None')

        # Test by passing FederatedDataset object
        # Do not use mock otherwise it will raise a type error
        td_expected = FederatedDataSet({'node-1': [{'dataset_id': 'ids'}]})
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data, td_expected, 'Setter for training data did not set given FederatedDataset '
                                                     'object')

        # Test by passing dict
        td_expected = {'node-1': [{'dataset_id': 'ids', 'test_ratio': .0}]}
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data.data(), td_expected, 'Setter for training data did not set given '
                                                            'FederatedDataset object')

        # Test by passing invalid type of training_data
        td_expected = 12
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_data(training_data=td_expected)

        # Test if the argument `from_tags` is not type of bool
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_data(training_data=None, from_tags=td_expected)

        # Test when job is not None
        self.mock_logger_debug.reset_mock()
        td_expected = {'node-1': [{'dataset_id': 'ids', 'test_ratio': .0}]}
        self.test_exp._job = MagicMock()
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data.data(), td_expected, 'Setter for training data did not set given '
                                                            'FederatedDataset object')
        self.assertEqual(self.mock_logger_debug.call_count, 3, "Logger debug is called unexpected times")

        # Test when secagg is not None
        self.mock_logger_debug.reset_mock()
        td_expected = {'node-1': [{'dataset_id': 'ids', 'test_ratio': .0}]}
        self.test_exp._secagg_servkey = MagicMock()
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data.data(), td_expected, 'Setter for training data did not set given '
                                                            'FederatedDataset object')
        self.assertEqual(self.mock_logger_debug.call_count, 4, "Logger debug is called unexpected times")

    def test_experiment_05_set_aggregator(self):
        """Testing setter for aggregator attribute of Experiment class"""

        # Set aggregator with None
        aggregator = self.test_exp.set_aggregator(aggregator=None)
        self.assertIsInstance(aggregator, FedAverage, 'Setter for aggregator did not set proper FedAverage instance')

        # Set aggregator with a built object
        agg_expected = FedAverage()
        aggregator = self.test_exp.set_aggregator(aggregator=agg_expected)
        self.assertEqual(aggregator, agg_expected, 'Setter for aggregator did not set given aggregator object')

        # Set aggregator with an instance
        agg_expected = FedAverage
        aggregator = self.test_exp.set_aggregator(aggregator=agg_expected)
        self.assertIsInstance(aggregator, FedAverage, 'Setter for aggregator did not set given aggregator instance')

        # Set aggregator that does not inherits base Aggregator class
        agg_expected = TestExperiment.fake_aggregator
        with self.assertRaises(SystemExit):
            self.test_exp.set_aggregator(aggregator=agg_expected)

        # Set aggregator that does not have proper type
        agg_expected = 13
        with self.assertRaises(SystemExit):
            self.test_exp.set_aggregator(aggregator=agg_expected)


    def test_experiment_06_set_strategy(self):
        """Testing setter for node_selection_strategy attribute of Experiment class"""

        # Test by passing strategy as None
        strategy = self.test_exp.set_strategy(node_selection_strategy=None)
        self.assertIsInstance(strategy, DefaultStrategy, 'Setter for strategy did not set proper '
                                                         'DefaultStrategy instance')

        # Test when ._fds is None
        self.test_exp.set_tags(None)
        self.test_exp.set_training_data(None)
        strategy = self.test_exp.set_strategy(None)
        self.assertIsNone(strategy, 'Strategy is not Nano where it should be')

        # Back to normal
        self.test_exp.set_tags(['tag-1', 'tag-2'])
        self.test_exp.set_training_data(None, from_tags=True)

        # Test by passing Strategy instance
        strategy_expected = DefaultStrategy
        strategy = self.test_exp.set_strategy(node_selection_strategy=strategy_expected)
        self.assertIsInstance(strategy, DefaultStrategy, 'Setter for strategy did not set proper '
                                                         'DefaultStrategy instance')

        # Test by passing built Strategy object
        fds = FederatedDataSetMock({})
        strategy_expected = DefaultStrategy(fds)
        strategy = self.test_exp.set_strategy(node_selection_strategy=strategy_expected)
        self.assertEqual(strategy, strategy_expected, 'Setter for strategy did not set expected strategy object')

        # Set strategy that does not inherit base Strategy class
        strategy_expected = TestExperiment.fake_strategy
        with self.assertRaises(SystemExit):
            self.test_exp.set_strategy(node_selection_strategy=strategy_expected)

        # Set strategy that is not in proper type
        strategy_expected = 13
        with self.assertRaises(SystemExit):
            self.test_exp.set_strategy(node_selection_strategy=strategy_expected)


    def test_experiment_07_set_round_limit(self):
        """Testing setter for round limit"""

        # Test setting round limit to None
        rl_expected = None
        round_limit = self.test_exp.set_round_limit(round_limit=rl_expected)
        self.assertEqual(round_limit, rl_expected, 'Setter for round limit did not set round_limit to None')

        # Test setting round limit less than current round
        self.mock_logger_error.reset_mock()
        self.test_exp._round_current = 2
        rl_expected = 1
        self.test_exp.set_round_limit(round_limit=rl_expected)
        self.mock_logger_error.assert_called_once()

        # Back to normal
        self.test_exp._round_current = 0

        # Test setting round_limit properly
        rl_expected = 1
        round_limit = self.test_exp.set_round_limit(round_limit=rl_expected)
        self.assertEqual(round_limit, rl_expected, 'Setter for round limit did not set round_limit to 1')

        # back to normal
        self.test_exp._round_current = 0
        self.test_exp.set_round_limit(round_limit=4)

        # Test passing invalid type for round_limit
        rl_expected = 'toto'
        with self.assertRaises(SystemExit):
            self.test_exp.set_round_limit(round_limit=rl_expected)

        # Test passing negative round_limit
        rl_expected = -2
        with self.assertRaises(SystemExit):
            self.test_exp.set_round_limit(round_limit=rl_expected)


    def test_experiment_08_private_set_round_current(self):
        """ Testing private method for setting round current for the experiment """

        # Test raise SystemExit when argument not in valid int type
        with self.assertRaises(SystemExit):
            self.test_exp._set_round_current('tot')

        # Test setting round current to negative
        with self.assertRaises(SystemExit):
            self.test_exp._set_round_current(-1)

        # Test setting round current more then round limit
        rc = self.test_exp.round_limit() + 1
        with self.assertRaises(SystemExit):
            self.test_exp._set_round_current(rc)

        # Test setting proper round current
        rcurrent_expected = 2
        rcurrent = self.test_exp._set_round_current(rcurrent_expected)
        self.assertEqual(rcurrent, rcurrent_expected, 'Setter for round current did not properly set the current round')


    def test_experiment_09_set_experimentation_folder(self):
        """ Test setting experimentation folder for the experiment """

        # Test passing None for folder path
        folder = self.test_exp.set_experimentation_folder(None)
        self.assertEqual(folder, 'Experiment_101', 'set_experimentation_folder did not properly create folder')

        # create_folder will return `Experiment_101` since it is mocked
        expected_folder = 'Experiment_101'
        folder = self.test_exp.set_experimentation_folder(expected_folder)
        self.assertEqual(folder, expected_folder, 'Folder for experiment is not set correctly')

        # Test raising SystemExit
        with self.assertRaises(SystemExit):
            self.test_exp.set_experimentation_folder(12)

        # Test warning
        with patch.object(fedbiomed.researcher.experiment, 'sanitize_filename'):
            self.mock_logger_warning.reset_mock()
            self.test_exp.set_experimentation_folder('test')
            self.mock_logger_warning.assert_called_once()

        # Test debug message when job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_experimentation_folder('12')
        self.mock_logger_debug.assert_called_once()


    def test_experiment_10_set_training_plan_class(self):
        """ Testing setter for training_plan  """

        # Test setting training_plan to None
        training_plan = self.test_exp.set_training_plan_class(None)
        self.assertIsNone(training_plan, 'Model class is not set as None')

        # Setting training_plan as string
        mc_expected = 'TestModel'
        training_plan = self.test_exp.set_training_plan_class(mc_expected)
        self.assertEqual(training_plan, mc_expected, 'Model class is not set properly while setting it in `str` type')

        # Back to normal
        self.test_exp._training_plan_path = None

        # Test by passing class
        training_plan = self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        self.assertEqual(training_plan, TestExperiment.FakeModelTorch,
                         'Model class is not set properly while setting it as class')

        # Test by passing class which has no subclass of one of the training plan
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_plan_class(FakeModel)

        # Back to normal
        self.test_exp._training_plan_path = None

        #  Test passing incorrect python identifier
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_plan_class('Fake Model')

        # Test passing class built object
        with self.assertRaises(SystemExit):
            training_plan = self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch())

        # Test passing class invalid type
        with self.assertRaises(SystemExit):
            training_plan = self.test_exp.set_training_plan_class(12)

        # Test passing class invalid type
        with self.assertRaises(SystemExit):
            training_plan = self.test_exp.set_training_plan_class({})

        # Test if ._job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_training_plan_class('FakeModel')
        # There will be two logger.debug call
        #  First    : Experiment is not fully configured since training_plan_path is still None
        #  Second   : Update Job since training_plan has changed
        self.assertEqual(self.mock_logger_debug.call_count, 2, 'Logger debug is called unexpected time while setting '
                                                               'model class')

    def test_experiment_11_set_training_plan_path(self):
        """ Testing setter for training_plan_path of experiment """

        # Test training_plan_path is None
        training_plan_path = self.test_exp.set_training_plan_path(None)
        self.assertIsNone(training_plan_path, 'Setter for training_plan_path did not set training_plan_path to None')

        # Test passing path for training_plan_file
        fake_training_plan_path = self.create_fake_training_plan_file('fake_model_2.py')
        training_plan_path = self.test_exp.set_training_plan_path(fake_training_plan_path)
        self.assertEqual(training_plan_path, fake_training_plan_path, 'Setter for training_plan_path did not set training_plan_path properly')

        # Test
        with patch.object(fedbiomed.researcher.experiment, 'sanitize_filepath') as m:
            m.return_value = 'test'
            with self.assertRaises(SystemExit):
                self.test_exp.set_training_plan_path(fake_training_plan_path)

        # Test invalid type of training_plan_path argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_plan_path(12)

        # Test when mode class is also set
        self.test_exp.set_training_plan_class('FakeModel')
        self.test_exp.set_training_plan_path(fake_training_plan_path)
        self.assertEqual(self.test_exp._training_plan_is_defined, True, '_training_plan_is_defined returns False even training_plan and '
                                                                'training_plan_path is fully configured')
        # Test if `._job` is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_training_plan_path(fake_training_plan_path)
        # There will be one debug call. If model_is_defined is False there might be two calls.
        # Since _training_plan_is_defined has become True with previous test block there will be only one call
        self.mock_logger_debug.assert_called_once()


    def test_experiment_12_set_model_arguments(self):
        """ Testing setter for model arguments of Experiment """

        # Test setting model_args as in invalid type
        with self.assertRaises(SystemExit):
            self.test_exp.set_model_args(None)

        # Test setting model_args properly with dict
        ma_expected = {'arg-1': 100}
        model_args = self.test_exp.set_model_args(ma_expected)
        self.assertDictEqual(ma_expected, model_args, 'Model arguments has not been set correctly by setter')

        # Test setting model_args while the ._job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        model_args = self.test_exp.set_model_args(ma_expected)
        # There will be one debug call.
        self.assertDictEqual(ma_expected, model_args, 'Model arguments has not been set correctly by setter')
        self.mock_logger_debug.assert_called_once()

    def test_experiment_13_set_training_arguments(self):
        """Testing setter for training arguments of Experiment """

        # Test setting model_args as in invalid type
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_args("this is not a dict")

        # Test setting model_args properly with dict
        ma_expected = {'batch_size': 25}
        train_args = self.test_exp.set_training_args(ma_expected)
        self.assertSubDictInDict(ma_expected, train_args, 'Training arguments has not been set correctly by setter')

        # test update of testing_args with argument `reset` set to False
        ma_expected_2 = {'batch_size': 10}
        train_args_2 = self.test_exp.set_training_args(ma_expected_2, reset=False)
        ma_expected_2.update(ma_expected_2)
        self.assertSubDictInDict(ma_expected_2, train_args_2)

        # test update of testing_args with argument `reset` set to True
        ma_expected_3 = {'batch_size': 1}
        train_args_3 = self.test_exp.set_training_args(ma_expected_3, reset=True)
        self.assertSubDictInDict(ma_expected_3, train_args_3)
        self.assertNotIn(list(ma_expected.keys()), list(train_args_3.keys()))
        self.assertNotIn(list(ma_expected_3.keys()), list(train_args_3.keys()))

        # Test setting model_args while the ._job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        train_args = self.test_exp.set_training_args(ma_expected)
        # There will be one debug call.
        self.assertSubDictInDict(ma_expected, train_args, 'Training arguments has not been set correctly by setter')
        #### ???? why ????
        # self.mock_logger_debug.assert_called_once()

        # Training arguments with testing arguments
        expected_train_args = {
            'optimizer_args': {},
            'test_ratio': 0.3,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'test_metric': 'ACCURACY',
            'test_metric_args': {}
        }
        train_args = self.test_exp.set_training_args(expected_train_args, reset=True)
        #self.assertDictEqual(train_args, expected_train_args)

        # cannot be checked ye with TrainingArgs
        # the validation_hook will be difficult to write, since
        # it may depend on the order of the keys
        # Raises error - can not set test metric argument without setting metric
        #expected_train_args = {
        #    'test_metric_args': {}
        #}
        #with self.assertRaises(SystemExit):
        #    self.test_exp.set_training_args(expected_train_args, reset=False)

        # Raises error since test_metric_args is not of type dict
        expected_train_args = {
            'test_metric': 'ACCURACY',
            'test_metric_args': 'test'
        }
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_args(expected_train_args, reset=False)

    def test_experiment_15_set_test_ratio(self):
        """
        Tests test_ratio setter `set_test_ratio`, correct uses and
        Exceptions
        """
        # case 1
        # add test_ratio when federated_dataset is not defined
        ratio_1_1 = .2
        self.test_exp.set_test_ratio(ratio_1_1)

        # get training data
        training_data_1 = self.test_exp.training_args()
        self.assertEqual(training_data_1['test_ratio'], ratio_1_1)

        # changing the value of `test_ratio`
        ratio_1_2 = .4

        self.test_exp.set_test_ratio(ratio_1_2)
        self.assertEqual(self.test_exp._training_args['test_ratio'], ratio_1_2)

        # case 2: setting a Job and a test_ratio afterwards
        self.test_exp._training_plan_is_defined = True
        self.test_exp.set_training_plan_class = TestExperiment.FakeModelTorch
        self.test_exp.set_job()
        ratio_2 = .8

        self.test_exp.set_test_ratio(ratio_2)

        self.assertEqual(self.test_exp._job._training_args['test_ratio'], ratio_2)

        # case 3: bad test_ratio values (triggers SystemExit exception)
        # 3.1 : test_ratio type is not correct
        # 3.2 : test_ratio is a float not whithin [0;1] interval
        ratio_3_1 = "some value"
        with self.assertRaises(SystemExit):
            self.test_exp.set_test_ratio(ratio_3_1)

        # check good interval values
        ratio_in  = 0.0
        ratio_out = self.test_exp.set_test_ratio(ratio_in)
        self.assertEqual(ratio_in, ratio_out)

        ratio_in  = 1.0
        ratio_out = self.test_exp.set_test_ratio(ratio_in)
        self.assertEqual(ratio_in, ratio_out)

        # check bad values
        for ratio in ( -1.0, -0.001, 1.0001, 2.0):
            with self.assertRaises(SystemExit):
                self.test_exp.set_test_ratio(ratio)


    @patch('fedbiomed.researcher.job.Job')
    @patch('fedbiomed.researcher.job.Job.__init__')
    def test_experiment_16_set_test_metric(self, mock_job_init, mock_job):
        """
        Tests testing metric setter `set_test_metric
        """

        # case 1. metric has been passed as a string
        metric_1 = "ACCURACY"
        metric_args_1 = {'normalize': True}

        self.test_exp.set_test_metric(metric=metric_1, **metric_args_1)

        training_args_1 = self.test_exp.training_args()

        self.assertEqual(training_args_1['test_metric'], metric_1)
        self.assertDictEqual(training_args_1['test_metric_args'], metric_args_1)
        # case 2. metric has been passed as a Enum / callable
        # TODO

        # case 3: failure, incorrect data type
        with self.assertRaises(SystemExit):
            self.test_exp.set_test_metric(True)

        # case 4: failure unsupported metric
        with self.assertRaises(SystemExit):
            self.test_exp.set_test_metric('ABBURACY')


        # case 4: Update jobs training arguments
        self.test_exp.set_job()
        self.test_exp.set_test_metric('ACCURACY')


    def test_experiment_17_set_test_on_global_updates(self):

        # Set job
        result = self.test_exp.set_test_on_global_updates(True)
        self.assertTrue(result)
        self.assertTrue(self.test_exp.training_args()['test_on_global_updates'])

        # Test wrong type
        with self.assertRaises(SystemExit):
            self.test_exp.set_test_on_global_updates('NotBool')


    def test_experiment_18_set_test_on_local_updates(self):

        # Set job
        result = self.test_exp.set_test_on_local_updates(True)
        self.assertTrue(result)
        self.assertTrue(self.test_exp.training_args()['test_on_local_updates'])
        # Test wrong type
        with self.assertRaises(SystemExit):
            self.test_exp.set_test_on_local_updates('NotBool')


    @patch('fedbiomed.researcher.job.Job')
    @patch('fedbiomed.researcher.job.Job.__init__')
    def test_experiment_19_set_job(self, mock_job_init, mock_job):
        """ Testing setter for Job in Experiment class """

        job_expected = "JOB"
        mock_job.return_value = job_expected
        mock_job_init.return_value = None

        # Test to override existing Job with set_job
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = True  # Assign any value to not make it None
        self.test_exp.set_job()
        # There will two logger.debug calls
        # First  : About Experiment Job has been changed
        # Second : Missing proper model definition
        self.assertEqual(self.mock_logger_debug.call_count, 2)

        # Back to normal
        self.test_exp._job = None

        # Test when ._fds is None
        self.test_exp._training_plan_is_defined = True
        self.test_exp._fds = None
        self.mock_logger_debug.reset_mock()
        self.test_exp.set_job()
        self.mock_logger_debug.assert_called_once()

        # Back to normal
        self.test_exp._fds = True  # Assign any value to not make it None

        # Test proper set job when everything is ready to create Job
        self.test_exp._training_plan_is_defined = True
        self.test_exp.set_training_plan_class = TestExperiment.FakeModelTorch
        job = self.test_exp.set_job()
        self.assertIsInstance(job, Job, 'Job has not been set properly')

    def test_experiment_20_set_save_breakpoints(self):
        """ Test setter for save_breakpoints attr of experiment class """

        # Test invalid type of argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_save_breakpoints(None)

        # test valid type of argument
        sb = self.test_exp.set_save_breakpoints(True)
        self.assertTrue(sb, 'save_breakpoint has not been set correctly')

    @patch('fedbiomed.researcher.experiment.Job')
    @patch('fedbiomed.researcher.experiment.SecaggServkeyContext')
    @patch('fedbiomed.researcher.experiment.SecaggBiprimeContext')
    def test_experiment_21_set_use_secagg(
        self,
        mock_secaggbiprimecontext,
        mock_secaggservkeycontext,
        mock_job,
    ):
        """ Test setter for use_secagg attr of experiment class """

        # Test invalid type of arguments
        use_secaggs = [ None, 3, 'toto', [True], {False} ]
        timeouts = [ None, 'titi', [2.4], {3.5}]
        for u in use_secaggs:
            for t in timeouts:
                with self.assertRaises(SystemExit):
                    self.test_exp.set_use_secagg(use_secagg=u)
                with self.assertRaises(SystemExit):
                    self.test_exp.set_use_secagg(timeout=t)

        # Test valid arguments + succeeds setting secagg context
        tags_cases = [
            None,
            ['tag1', 'tag2'],
        ]
        parties = ['party1', 'party2', 'party3', 'party4']
        job_id = 'my_test_job_id'

        class FakeJob:
            def __init__(self):
                self.id = job_id

        for tags in tags_cases:
            exp = Experiment(tags=tags)
            mock_secaggservkeycontext.return_value = FakeSecaggServkeyContext(parties, job_id)
            mock_secaggbiprimecontext.return_value = FakeSecaggBiprimeContext(parties)
            mock_job.return_value = FakeJob()
            # we should not set directly exp._job (internal to exp) for unit tests
            # but ...
            exp._job = mock_job

            use_false = exp.set_use_secagg(False)
            context_false_servkey, context_false_biprime = exp.secagg_context()
            use_true = exp.set_use_secagg(True)
            context_true_servkey, context_true_biprime = exp.secagg_context()

            self.assertFalse(use_false)
            self.assertEqual(context_false_servkey, None)
            self.assertEqual(context_false_biprime, None)
            self.assertTrue(use_true)
            self.assertEqual(context_true_servkey.cont, FAKE_CONTEXT_VALUE)
            self.assertEqual(context_true_biprime.cont, FAKE_CONTEXT_VALUE)

        # Test valid arguments + fails setting secagg context
        tags_cases = [
            None,
            ['tag1', 'tag2'],
        ]
        parties = ['party1', 'party2', 'party3', 'party4']
        job_id = 'my_test_job_id'
        setup_results = [
            [True, False],
            [False, True],
            [False, False]
        ]

        for tags in tags_cases:
            for result in setup_results:
                exp = Experiment(tags=tags)
                mock_secaggservkeycontext.return_value = FakeSecaggServkeyContext(parties, job_id)
                mock_secaggbiprimecontext.return_value = FakeSecaggBiprimeContext(parties)
                mock_secaggservkeycontext.return_value.set_setup_success(result[0])
                mock_secaggbiprimecontext.return_value.set_setup_success(result[1])
                mock_job.return_value = FakeJob()
                # we should not set directly exp._job (internal to exp) for unit tests
                # but ...
                exp._job = mock_job

                use_false = exp.set_use_secagg(False)
                context_false_servkey, context_false_biprime = exp.secagg_context()
                use_true = exp.set_use_secagg(True)
                context_true_servkey, context_true_biprime = exp.secagg_context()

                self.assertFalse(use_false)
                self.assertEqual(context_false_servkey, None)
                self.assertEqual(context_false_biprime, None)
                self.assertFalse(use_true)
                self.assertTrue(context_true_servkey.cont is None or context_true_biprime.cont is None)



    def test_experiment_22_set_tensorboard(self):
        """ Test setter for tensorboard """

        # Test invalid type of argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_tensorboard(None)

        # test valid type of argument
        sb = self.test_exp.set_tensorboard(True)
        self.assertTrue(sb, 'tensorboard has not been set correctly')

        # test valid type of argument
        sb = self.test_exp.set_tensorboard(False)
        self.assertFalse(sb, 'tensorboard has not been set correctly')


    @patch('fedbiomed.researcher.experiment.Experiment.breakpoint')
    @patch('fedbiomed.researcher.aggregators.fedavg.FedAverage.aggregate')
    @patch('fedbiomed.researcher.aggregators.Aggregator.create_aggregator_args')
    @patch('fedbiomed.researcher.strategies.default_strategy.DefaultStrategy.refine')
    @patch('fedbiomed.researcher.job.Job.training_plan', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.training_replies', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.start_nodes_training_round')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    @patch('fedbiomed.researcher.job.Job.__init__')
    def test_experiment_23_run_once(self,
                                    mock_job_init,
                                    mock_job_updates_params,
                                    mock_job_training,
                                    mock_job_training_replies,
                                    mock_job_training_plan_type,
                                    mock_strategy_refine,
                                    mock_fedavg_create_aggregator_args,
                                    mock_fedavg_aggregate,
                                    mock_experiment_breakpoint):
        """ Testing run_once method of Experiment class """
        training_plan = MagicMock()
        training_plan.type = MagicMock()
        mock_job_init.return_value = None
        mock_job_training.return_value = None
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 'reply'}
        mock_job_training_plan_type.return_value = PropertyMock(return_value=training_plan)
        mock_strategy_refine.return_value = ({'param': 1}, [12.2])
        mock_fedavg_aggregate.return_value = None
        mock_fedavg_create_aggregator_args.return_value = ({}, {})
        mock_job_updates_params.return_value = "path/to/my/file", "http://some/url/to/my/file"
        mock_experiment_breakpoint.return_value = None

        # Test invalid `increase` arguments
        with self.assertRaises(SystemExit):
            self.test_exp.run_once(1)
        with self.assertRaises(SystemExit):
            self.test_exp.run_once("1")

        # Test when ._job is None
        with self.assertRaises(SystemExit):
            self.test_exp.run_once()

        # Test when strategy is None
        with self.assertRaises(SystemExit):
            self.test_exp._node_selection_strategy = None
            self.test_exp.run_once()

        # Test run_once when everything is ready to run

        # Set model class to be able to create Job
        self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        # Set default Job
        self.test_exp.set_job()
        # Set strategy again (it was removed)
        self.test_exp.set_strategy(None)
        
        result = self.test_exp.run_once()
        self.assertEqual(result, 1, "run_once did not successfully run the round")
        mock_job_training.assert_called_once()
        mock_strategy_refine.assert_called_once()
        mock_fedavg_aggregate.assert_called_once()
        mock_job_updates_params.assert_called_once()
        mock_experiment_breakpoint.assert_called_once()

        # Test the scenario where round_limit is reached
        self.test_exp.set_round_limit(1)
        # run once should log warning message
        self.mock_logger_warning.reset_mock()
        result = self.test_exp.run_once()
        self.assertEqual(result, 0)
        self.mock_logger_warning.assert_called_once()

        # Update training_replies mock value since round_current has been increased
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 'reply'}

        # Try same scenario with increase argument as True
        round_limit = self.test_exp.round_limit()
        result = self.test_exp.run_once(increase=True)
        self.assertEqual(result, 1, 'Run once did not run the training where it should')
        self.assertEqual(self.test_exp.round_limit(), round_limit + 1,
                         'Round limit has not been increased after running run_once with '
                         'increase=True')

        # Try same scenario with test_after argument as True

        # resetting mocks
        mock_job_training.reset_mock()
        mock_strategy_refine.reset_mock()
        mock_fedavg_aggregate.reset_mock()
        mock_job_updates_params.reset_mock()
        mock_experiment_breakpoint.reset_mock()
        # action
        self.test_exp._round_current = 1
        result = self.test_exp.run_once(test_after=True)
        # testing calls
        mock_strategy_refine.assert_called_once()
        mock_fedavg_aggregate.assert_called_once()
        mock_job_updates_params.assert_called_once()
        mock_experiment_breakpoint.assert_called_once()
        self.assertEqual(mock_job_training.call_count, 2)
        # additional checks
        self.assertEqual(result, 1)
    
    @patch('fedbiomed.researcher.experiment.Experiment.breakpoint')
    @patch('fedbiomed.researcher.aggregators.scaffold.Scaffold.aggregate')
    @patch('fedbiomed.researcher.aggregators.scaffold.Scaffold.create_aggregator_args')
    @patch('fedbiomed.researcher.strategies.default_strategy.DefaultStrategy.refine')
    @patch('fedbiomed.researcher.job.Job.training_plan', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.training_replies', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.start_nodes_training_round')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    @patch('fedbiomed.researcher.job.Job.__init__')  
    def test_experiment_23_run_once_with_scaffold_and_training_args(self,
                                                                    mock_job_init,
                                                                    mock_job_updates_params,
                                                                    mock_job_training,
                                                                    mock_job_training_replies,
                                                                    mock_job_training_plan_type,
                                                                    mock_strategy_refine,
                                                                    mock_scaffold_create_aggregator_args,
                                                                    mock_scaffold_aggregate,
                                                                    mock_experiment_breakpoint):
        # try test with specific training_args
        # related to regression due to Scaffold introduction applied on MedicalFolderDataset
        training_plan = MagicMock()
        training_plan.type = MagicMock()
        mock_job_init.return_value = None
        mock_job_training.return_value = None
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 'reply'}
        mock_job_training_plan_type.return_value = PropertyMock(return_value=training_plan)
        mock_strategy_refine.return_value = ({'param': 1}, [12.2])
        mock_scaffold_aggregate.return_value = None
        mock_scaffold_create_aggregator_args.return_value = ({}, {})
        mock_job_updates_params.return_value = "path/to/my/file", "http://some/url/to/my/file"
        mock_experiment_breakpoint.return_value = None

        # Set model class to be able to create Job
        self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        # Set default Job
        self.test_exp.set_job()
        # Set strategy
        self.test_exp.set_strategy(None)
        # set training_args
        self.test_exp.set_training_args({'num_updates': 1000})
        # set Scaffold aggregator
        self.test_exp.set_aggregator(Scaffold(server_lr=.1))

        result = self.test_exp.run_once()
        self.assertEqual(result, 1, "run_once did not successfully run the round")

    @patch('fedbiomed.researcher.aggregators.fedavg.FedAverage.aggregate')
    @patch('fedbiomed.researcher.job.Job.training_plan', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.training_replies', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.start_nodes_training_round')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    @patch('fedbiomed.researcher.job.Job.__init__')  
    def test_experiment_24_strategy(self,
                                    mock_job_init,
                                    mock_job_updates_params,
                                    mock_job_training,
                                    mock_job_training_replies,
                                    mock_job_training_plan_type,
                                    mock_fedavg_aggregate):
        """test_experiment_24_strategy: testing several case where strategy may fail"""
        # FIXME: this is more of an integration test than a unit test
        
        # set up:
        model_param = [1, 2, 3]
        num_updates = 1000
        node_ids = ['node-1', 'node-2']
        node_sample_size = [10, 20]  # size of samples parsedby each node
        
        assert len(node_ids) == len(node_sample_size), "wrong setup for test: node_ids and node_sample_size should be" \
            "of the same size"
        
        training_plan = MagicMock()
        training_plan.type = MagicMock()
        # mocking job 
        mock_job_init.return_value = None
        mock_job_training.return_value = None
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 
            Responses( [{ 'success': True,
                         'msg': "this is a sucessful training",
                             'dataset_id': 'dataset-id-123abc',
                             'node_id': node_id,
                             'params_path': '/path/to/my/file',
                             'params': model_param,
                             'sample_size': sample_size
                             } for node_id, sample_size in zip(node_ids, node_sample_size)
                        ])}
        mock_job_training_plan_type.return_value = PropertyMock(return_value=training_plan)
        mock_job_updates_params.return_value = "path/to/my/file", "http://some/url/to/my/file"
        
        # mocking aggregator
        mock_fedavg_aggregate.return_value = None

        # disable patchers (enabled in the test set up)
        for _patch in self.patchers:
            _patch.stop()
        # Set model class to be able to create Job
        self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        # Set default Job
        self.test_exp.set_job()
        # Set strategy
        self.test_exp.set_strategy(DefaultStrategy(data=FederatedDataSet({
                                                    'node-1': [{'dataset_id': 'dataset-id-1',
                                                                'shape': [100, 100]}],
                                                    'node-2': [{'dataset_id': 'dataset-id-2',
                                                                'shape': [120, 120], 
                                                                'test_ratio': .0}],
                                                })))
        # set training_args
        self.test_exp.set_training_args({'num_updates': num_updates})
        self.test_exp.set_aggregator(FedAverage())
        # removing breakpoints (otherwise test will fail)
        self.test_exp.set_save_breakpoints(False)
        result = self.test_exp.run_once()
        
        weigths = {node_id: sample_size/sum(node_sample_size) for node_id, sample_size in zip(node_ids,
                                                                                              node_sample_size)}
        model_params = {node_id: model_param for node_id in node_ids}
        mock_fedavg_aggregate.assert_called_with(model_params, weigths,
                                                 global_model=unittest.mock.ANY,
                                                 training_plan=unittest.mock.ANY,
                                                 training_replies=mock_job_training_replies(),
                                                 node_ids=node_ids,
                                                 n_updates=num_updates,
                                                 n_round=0,)
        
        # repeat experiment but with a wrong sample_size
        
        node_sample_size = [10, None]
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 
            Responses( [{ 'success': True,
                         'msg': "this is a sucessful training",
                             'dataset_id': 'dataset-id-123abc',
                             'node_id': node_id,
                             'params_path': '/path/to/my/file',
                             'params': model_param,
                             'sample_size': sample_size
                             } for node_id, sample_size in zip(node_ids, node_sample_size)
                        ])}
        
        with self.assertRaises(SystemExit):
            # should raise a FedbiomedStrategyError, describing the error
            self.test_exp.run_once()

    @patch('fedbiomed.researcher.experiment.Experiment.run_once')
    def test_experiment_24_run(self, mock_exp_run_once):
        """ Testing run method of Experiment class """

        def run_once_side_effect(increase, test_after=False):
            inc = self.test_exp.round_current() + 1
            self.test_exp._set_round_current(inc)
            return 1

        mock_exp_run_once.side_effect = run_once_side_effect

        # Test if rounds is not `int`
        with self.assertRaises(SystemExit):
            self.test_exp.run(rounds='not-int')

        # Test if round is less than 1
        with self.assertRaises(SystemExit):
            self.test_exp.run(rounds=0)

        # Test if increase is not bool
        with self.assertRaises(SystemExit):
            self.test_exp.run(rounds=2, increase='not-bool')

        # Test run() without passing any argument
        expected_rounds = self.test_exp.round_limit()
        print(expected_rounds)
        rounds = self.test_exp.run()
        self.assertEqual(rounds, expected_rounds, 'Rounds  result of run() function do not match with round limit')

        # Test run if all rounds has been completed
        # (round hsa been completed in the previous exp.run())
        self.mock_logger_warning.reset_mock()
        rounds = self.test_exp.run()
        self.assertEqual(rounds, 0, f'run() returned {rounds} where the expected value is 0')
        # Logger warning should be called once to inform user about the round
        # limit has been reached
        self.mock_logger_warning.assert_called_once()

        # Test run by passing rounds lees than round limit
        # Since increase is True by default round_limit will increase
        round_limit_before = self.test_exp.round_limit()
        expected_rounds = 2
        rounds = self.test_exp.run(rounds=expected_rounds, increase=True)
        self.assertEqual(rounds, 2, 'Rounds  result of run() function do not match with round limit')
        self.assertEqual(self.test_exp.round_limit(),
                         round_limit_before + expected_rounds,
                         'run() function did not update round limit when round limit is reach and increase is True')

        # Test run by passing rounds more than round_limit but with increase=False
        self.mock_logger_warning.reset_mock()
        expected_rounds = self.test_exp.round_limit() + 1
        rounds = self.test_exp.run(rounds=expected_rounds, increase=False)
        self.assertEqual(rounds, 0, f'run() returned {rounds} where the expected value is 0')
        self.mock_logger_warning.assert_called_once()

        # Test reducing the number of rounds to run in the experiment
        self.mock_logger_warning.reset_mock()
        self.test_exp.set_round_limit(self.test_exp.round_limit() + 1)
        expected_rounds = self.test_exp.round_limit() + 1
        rounds = self.test_exp.run(rounds=expected_rounds, increase=False)
        self.assertEqual(rounds, 1, f'run() returned {rounds} where the expected value is 0')
        # Logger.warning will inform user about the number of rounds will be reduced
        self.mock_logger_warning.assert_called_once()

        # Test if _round_limit is not int
        self.test_exp.set_round_limit(None)
        self.mock_logger_warning.reset_mock()
        rounds = self.test_exp.run()
        self.assertEqual(rounds, 0, f'run() returned {rounds} where the expected value is 0')
        # Logger warning should be called once to inform user about the round
        # limit should be set
        self.mock_logger_warning.assert_called_once()

        # Test the case where run_once returns 0
        mock_exp_run_once.side_effect = None
        mock_exp_run_once.return_value = 0
        with self.assertRaises(SystemExit):
            self.test_exp.run(rounds=1)

        mock_exp_run_once.reset_mock()
        mock_exp_run_once.return_value = 1
        # Test run when testing is activated for global updates
        self.test_exp.set_round_limit(self.test_exp.round_current() + 1)
        self.test_exp.set_test_on_global_updates(True)
        rounds = self.test_exp.run()
        self.assertEqual(rounds, 1)

    @patch('builtins.open')
    @patch('fedbiomed.researcher.job.Job.training_plan_file', new_callable=PropertyMock)
    def test_experiment_25_training_plan_file(self,
                                      mock_training_plan_file,
                                      mock_open):
        """ Testing getter training_plan_file of the experiment class """

        m_open = MagicMock()
        m_open.read = MagicMock(return_value=None)
        m_open.close.return_value = None

        mock_open.return_value = m_open
        mock_training_plan_file.return_value = 'path/to/model'

        # Test if ._job is not defined
        with self.assertRaises(SystemExit):
            self.test_exp.training_plan_file()

        # Test if display is not bool
        with self.assertRaises(SystemExit):
            self.test_exp.training_plan_file(display='not-bool')

        # Test when display is false
        self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        self.test_exp.set_job()
        result = self.test_exp.training_plan_file(display=False)
        self.assertEqual(result,
                         'path/to/model',
                         f'training_plan_file() returned {result} where it should have returned `path/to/model`')

        # Test when display is true
        result = self.test_exp.training_plan_file(display=True)
        self.assertEqual(result,
                         'path/to/model',
                         f'training_plan_file() returned {result} where it should have returned `path/to/model`')

        # Test if `open()` raises OSError
        mock_open.side_effect = OSError
        with self.assertRaises(SystemExit):
            result = self.test_exp.training_plan_file(display=True)

    @patch('fedbiomed.researcher.job.Job.__init__', return_value=None)
    @patch('fedbiomed.researcher.job.Job.check_training_plan_is_approved_by_nodes')
    def test_experiment_26_check_training_plan_status(self,
                                              mock_job_model_is_approved,
                                              mock_job):
        """Testing method that checks model status """

        # Test error if ._job is not defined
        with self.assertRaises(SystemExit):
            self.test_exp.check_training_plan_status()

        # Test when job is defined
        expected_approved_result = {'node-1': {'is_approved': False}}
        mock_job_model_is_approved.return_value = expected_approved_result
        self.test_exp.set_training_plan_class(TestExperiment.FakeModelTorch)
        self.test_exp.set_job()
        result = self.test_exp.check_training_plan_status()
        self.assertDictEqual(result, expected_approved_result, 'check_training_plan_status did not return expected value')

    def test_experiment_27_breakpoint_raises(self):
        """ Testing the scenarios where the method breakpoint() raises error """

        # Test if self._round_current is less than 1
        with self.assertRaises(SystemExit):
            self.test_exp.breakpoint()

        # Test if ._fds is None
        self.test_exp._set_round_current(1)
        self.test_exp._fds = None
        with self.assertRaises(SystemExit):
            self.test_exp.breakpoint()

        # Back to normal fds
        self.test_exp.set_training_data(None, from_tags=True)

        # Test if strategy is None
        self.test_exp._node_selection_strategy = None
        with self.assertRaises(SystemExit):
            self.test_exp.breakpoint()

        # Back to normal strategy
        self.test_exp.set_strategy(None)

        # Test if _job is None (it is already None by default)
        with self.assertRaises(SystemExit):
            self.test_exp.breakpoint()


    @patch('fedbiomed.researcher.experiment.create_unique_file_link')
    @patch('fedbiomed.researcher.experiment.create_unique_link')
    @patch('fedbiomed.researcher.experiment.choose_bkpt_file')
    # testing _save_breakpoint + _save_aggregated_params
    # (not exactly a unit test, but probably more interesting)
    def test_experiment_28_save_breakpoint(
            self,
            patch_choose_bkpt_file,
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
        training_data = {'node1': [{'name': 'dataset1', 'test_ratio': .0}],
                         'node2': [{'name': 'dataset2', 'test_ratio': .0}]}
        # we want to test with non null values
        training_args = TrainingArgs( only_required = False )
        self.test_exp._training_args = training_args
        model_args = {'modelarg1': 'value1', 'modelarg2': 234, 'modelarg3': False}
        self.test_exp._model_args = model_args
        training_plan_file = '/path/to/my/training_plan_file.py'
        training_plan_class = 'MyOwnTrainingPlan'
        round_current = 2
        aggregator_state = {'aggparam1': 'param_value', 'aggparam2': 987, 'aggparam3': True}
        strategy_state = {'stratparam1': False, 'stratparam2': 'my_strategy', 'aggparam3': 0.45}
        job_state = {'jobparam1': {'sub1': 1, 'sub2': 'two'}, 'jobparam2': 'myjob_value'}
        use_secagg = [
            False,
            True
        ]
        secagg_servkey_context = [
            None,
            {'param1': 'val1', 'param2': 34}
        ]
        secagg_biprime_context = [
            None,
            {'biprime1': False, 'biprime2': 'myval'}
        ]

        # aggregated_params
        agg_params = {
            'entry1': {'params_path': '/dummy/path/to/aggparams/params_path.pt'},
            'entry2': {'params_path': '/yet/another/path/other_params_path.pt'}
        }
        self.test_exp._aggregated_params = agg_params


        # patch choose_bkpt_file create_unique_{file_}link  with minimal functions
        def side_bkpt_file(exp_folder, round):
            # save directly in experiment folder to avoid creating additional dirs
            return self.experimentation_folder_path, bkpt_file

        patch_choose_bkpt_file.side_effect = side_bkpt_file


        def side_create_ul(bkpt_folder_path, link_src_prefix, link_src_postfix, link_target_path):
            return os.path.join(bkpt_folder_path, link_src_prefix + link_src_postfix)

        patch_create_ul.side_effect = side_create_ul


        def side_create_ufl(bkpt_folder_path, file_path):
            return os.path.join(bkpt_folder_path, os.path.basename(file_path))

        patch_create_ufl.side_effect = side_create_ufl

        # build minimal objects, needed to extract state by calling object method
        # (cannot just patch a method of a non-existing object)
        class Aggregator():
            def save_state(self, training_plan: BaseTrainingPlan,
                           breakpoint_path: str, **kwargs):
                return aggregator_state

        self.test_exp._aggregator = Aggregator()

        class Strategy():
            def save_state(self):
                return strategy_state

        self.test_exp._node_selection_strategy = Strategy()

        class SecaggContext():
            pass

        class SecaggServkeyContext(SecaggContext):
            def save_state(self):
                return secagg_servkey_context[1]

        secagg_servkey = [
            None,
            SecaggServkeyContext()
        ]

        class SecaggBiprimeContext(SecaggContext):
            def save_state(self):
                return secagg_biprime_context[1]

        secagg_biprime = [
            None,
            SecaggBiprimeContext()
        ]

        # use the mocked FederatedDataSet
        self.test_exp._fds = FederatedDataSet(training_data)

        # could also do: self.test_exp._set_round_current(round_current)
        self.test_exp._round_current = round_current

        # build minimal Job object
        class DummyJob():
            def __init__(self):
                self._training_plan = None
            def save_state(self, breakpoint_path):
                return job_state
            
            @property
            def training_plan(self):
                return self._training_plan

        self.test_exp._job = DummyJob()
        # patch Job training_plan / training_plan_file
        self.test_exp._job.training_plan_file = training_plan_file
        self.test_exp._job.training_plan_name = training_plan_class


        for secagg_i in range(2):
            self.test_exp._use_secagg = use_secagg[secagg_i]
            self.test_exp._secagg_servkey = secagg_servkey[secagg_i]
            self.test_exp._secagg_biprime = secagg_biprime[secagg_i]

            # action
            patcher_secagg_context = patch('fedbiomed.researcher.experiment.SecaggContext', SecaggContext)
            patcher_secagg_context.start()
            self.test_exp.breakpoint()
            patcher_secagg_context.start()

            # verification
            final_training_plan_path = os.path.join(
                self.experimentation_folder_path,
                'model_' + str("{:04d}".format(round_current - 1)) + '.py')
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
            self.assertEqual(final_state['training_args'], training_args.dict())
            self.assertEqual(final_state['model_args'], model_args)
            self.assertEqual(final_state['training_plan_path'], final_training_plan_path)
            self.assertEqual(final_state['training_plan_class'], training_plan_class)
            self.assertEqual(final_state['round_current'], round_current)
            self.assertEqual(final_state['round_limit'], self.round_limit)
            self.assertEqual(final_state['experimentation_folder'], self.experimentation_folder)
            self.assertEqual(final_state['aggregator'], aggregator_state)
            self.assertEqual(final_state['node_selection_strategy'], strategy_state)
            self.assertEqual(final_state['tags'], self.tags)
            self.assertEqual(final_state['aggregated_params'], final_agg_params)
            self.assertEqual(final_state['job'], job_state)
            self.assertEqual(final_state['use_secagg'], use_secagg[secagg_i])
            self.assertEqual(final_state['secagg_servkey'], secagg_servkey_context[secagg_i])
            self.assertEqual(final_state['secagg_biprime'], secagg_biprime_context[secagg_i])

        # Test errors while writing brkp json file
        with patch.object(fedbiomed.researcher.experiment, 'open') as m:
            m.side_effect = OSError
            with self.assertRaises(SystemExit):
                self.test_exp.breakpoint()

        with patch.object(fedbiomed.researcher.experiment.json, 'dump') as m:
            m.side_effect = OSError
            with self.assertRaises(SystemExit):
                self.test_exp.breakpoint()

            m.side_effect = TypeError
            with self.assertRaises(SystemExit):
                self.test_exp.breakpoint()

            m.side_effect = RecursionError
            with self.assertRaises(SystemExit):
                self.test_exp.breakpoint()


    @patch('fedbiomed.researcher.experiment.Experiment.training_plan')
    #@patch('fedbiomed.researcher.experiment.Experiment._create_object')
    @patch('fedbiomed.researcher.experiment.find_breakpoint_path')
    # test load_breakpoint + _load_aggregated_params
    # cannot test Experiment constructor, need to fake it
    # (not exactly a unit test, but probably more interesting)
    def test_experiment_29_static_load_breakpoint(self,
                                                  patch_find_breakpoint_path,
                                                  #patch_create_object,
                                                  patch_training_plan
                                                  ):
        """ test `load_breakpoint` :
            1. if breakpoint file is json loadable
            2. if experiment is correctly configured from breakpoint
        """

        # Prepare breakpoint data
        bkpt_file = 'file_4_breakpoint'

        training_data = {'train_node1': [{'name': 'my_first_dataset', 2: 243}]}
        training_args = TrainingArgs( only_required = False )
        model_args = {'modarg1': True, 'modarg2': 7.12, 'modarg3': 'model_param_foo'}
        training_plan_path = '/path/to/breakpoint_training_plan_file.py'
        training_plan_class = 'ThisIsTheTrainingPlan'
        round_current = 1
        experimentation_folder = 'My_experiment_folder_258'
        aggregator_params = {'aggregator_name': 'dummy-aggregator',
                      'aggreg1': False, 'aggreg2': 'dummy_agg_param', 18: 'agg_param18'}
        strategy_params = {'strat1': 'test_strat_param', 'strat2': 421, 3: 'strat_param3'}
        aggregated_params = {
            '1': {'params_path': '/path/to/my/params_path_1.pt'},
            2: {'params_path': '/path/to/my/params_path_2.pt'}
        }
        job = {1: 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999}
        use_secagg = True
        secagg_servkey = {'secagg_id': '1234',
                          'researcher_id': '1234',
                          'status': True, 
                          'context': None,
                          'servkey1': 'A VALUE', 2: 247, 'parties': ['one', 'two'],
                          'job_id': 'A JOB1 ID',
                          'class': 'FakeSecaggServkeyContext',
                          'module': self.__module__}
        secagg_biprime = {'biprime1': 'ANOTHER VALUE', 'bip': 'rhyme', 'parties': ['three', 'four'], 'job_id': 'A JOB2 ID',
                          'class': 'FakeSecaggBiprimeContext', 'module': self.__module__}


        fake_aggregator = FakeAggregator()
        fake_aggregator._aggregator_args = aggregator_params
        
        fake_strategy = FakeStrategy(data=training_args)
        fake_strategy._parameters = strategy_params
        # breakpoint structure
        state = {
            'training_data': training_data,
            'training_args': training_args.dict(),
            'model_args': model_args,
            'training_plan_path': training_plan_path,
            'training_plan_class': training_plan_class,
            'round_current': round_current,
            'round_limit': self.round_limit,
            'experimentation_folder': experimentation_folder,
            'aggregator': {
                            "class": 'FakeAggregator',
                            "module": self.__module__,
                            "parameters": aggregator_params
                        },
            'node_selection_strategy': {
                            "class": 'FakeStrategy',
                            "module": self.__module__,
                            "parameters": strategy_params,
                            "fds": training_data
                        },
            'tags': self.tags,
            'aggregated_params': aggregated_params,
            'job': job,
            'use_secagg': use_secagg,
            'secagg_servkey': secagg_servkey,
            'secagg_biprime': secagg_biprime,
        }
        # create breakpoint file
        with open(os.path.join(self.experimentation_folder_path, bkpt_file), "w") as f:
            json.dump(state, f)

        # patch functions for loading breakpoint
        patch_find_breakpoint_path.return_value = self.experimentation_folder_path, bkpt_file

        # mocked model params
        model_params = {'something.bias': [12, 14], 'something.weight': [13, 15]}

        # target aggregated params
        final_aggregated_params = {
            1: {'params_path': '/path/to/my/params_path_1.pt'},
            2: {'params_path': '/path/to/my/params_path_2.pt'}
        }
        for aggpar in final_aggregated_params.values():
            aggpar['params'] = model_params
        # target breakpoint element arguments
        final_tags = self.tags
        final_experimentation_folder = experimentation_folder
        final_training_data = {'train_node1': [{'name': 'my_first_dataset',
                                                '2': 243}]}

        final_training_args = TrainingArgs(only_required=False)
        final_aggregator = {'aggregator_name': 'dummy-aggregator',
                            'aggreg1': False, 'aggreg2': 'dummy_agg_param', '18': 'agg_param18'}
        final_strategy = {'strat1': 'test_strat_param', 'strat2': 421, '3': 'strat_param3'}
        final_job = {'1': 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999}
        final_use_secagg = True
        final_secagg_servkey = {'servkey1': 'A VALUE', '2': 247, 'parties': ['one', 'two'], 'job_id': 'A JOB1 ID',
                                }
        final_secagg_biprime = {'biprime1': 'ANOTHER VALUE', 'bip': 'rhyme', 'parties': ['three', 'four'], 'job_id': 'A JOB2 ID',
                                'class': 'FakeSecaggBiprimeContext', 'module': self.__module__}

        
        # def side_create_object(args, **kwargs):
        #     return args

        # patch_create_object.side_effect = side_create_object

        class FakeModelInstance:
            def load(self, aggreg, to_params):
                return model_params

        patch_training_plan.return_value = FakeModelInstance()

        # could not have it working with a decorator or by patching the whole class
        # (we are in a special case : constructor of
        # an object instantiated from the `cls` of a class function)
        patches_experiment = [
            patch('fedbiomed.researcher.experiment.Experiment.__init__',
                  ExperimentMock.__init__),
            patch('fedbiomed.researcher.experiment.Experiment._set_round_current',
                  ExperimentMock._set_round_current)
        ]
        for p in patches_experiment:
            p.start()

        # Action - Tests

        # Test if breakpoint argument is not type of `str`
        with self.assertRaises(SystemExit):
            Experiment.load_breakpoint(breakpoint_folder_path=True)  # Not str

        # Test if open `open`  and json.load returns exception
        with patch.object(fedbiomed.researcher.experiment, 'open') as m_open, \
                patch.object(fedbiomed.researcher.experiment.json, 'load') as m_load:

            m_load = MagicMock()
            m_open.side_effect = OSError
            with self.assertRaises(SystemExit):
                Experiment.load_breakpoint(self.experimentation_folder_path)

            m_open.side_effect = None
            m_load.side_effect = json.JSONDecodeError
            with self.assertRaises(SystemExit):
                Experiment.load_breakpoint(self.experimentation_folder_path)

        # Test if model instance is None
        with patch.object(fedbiomed.researcher.experiment.Experiment, 'training_plan') as m_mi:
            m_mi.return_value = None
            with self.assertRaises(SystemExit):
                Experiment.load_breakpoint(self.experimentation_folder_path)

        # Test when everything is OK

        loaded_exp = Experiment.load_breakpoint(self.experimentation_folder_path)

        for p in patches_experiment:
            p.stop()

        # verification
        self.assertTrue(isinstance(loaded_exp, Experiment))
        self.assertEqual(loaded_exp._tags, final_tags)
        self.assertEqual(loaded_exp._fds.data(), final_training_data)
        self.assertEqual(loaded_exp._aggregator._aggregator_args, final_aggregator)
        self.assertIsInstance(loaded_exp._aggregator, Aggregator)
        self.assertEqual(loaded_exp._node_selection_strategy._parameters, final_strategy)
        self.assertIsInstance(loaded_exp._node_selection_strategy, Strategy)
        self.assertEqual(loaded_exp._round_current, round_current)
        self.assertEqual(loaded_exp._round_limit, self.round_limit)
        self.assertEqual(loaded_exp._experimentation_folder, final_experimentation_folder)
        self.assertEqual(loaded_exp._training_plan_class, training_plan_class)
        self.assertEqual(loaded_exp._training_plan_path, training_plan_path)
        self.assertEqual(loaded_exp._model_args, model_args)
        self.assertDictEqual(loaded_exp._training_args.dict(), final_training_args.dict())
        self.assertEqual(loaded_exp._job._saved_state, final_job)
        self.assertEqual(loaded_exp._aggregated_params, final_aggregated_params)
        self.assertTrue(loaded_exp._save_breakpoints)
        self.assertFalse(loaded_exp._monitor)
        self.assertEqual(loaded_exp._use_secagg, final_use_secagg)
        self.assertEqual(loaded_exp._secagg_servkey.parties, final_secagg_servkey['parties'])
        self.assertEqual(loaded_exp._secagg_biprime.parties, final_secagg_biprime['parties'])

    @patch('fedbiomed.researcher.experiment.create_unique_file_link')
    def test_experiment_30_static_save_aggregated_params(self,
                                                         mock_create_unique_file_link):
        """Testing static private method of experiment for saving aggregated params"""

        mock_create_unique_file_link.return_value = '/test/path/'

        # Test invalid type of arguments
        agg_params = 12  #
        with self.assertRaises(SystemExit):
            Experiment._save_aggregated_params(aggregated_params_init=agg_params, breakpoint_path="./")
        with self.assertRaises(SystemExit):
            Experiment._save_aggregated_params(aggregated_params_init={}, breakpoint_path=True)

        # Test if aggregated_params_init is dict but values not
        agg_params = {"node-1": True, "node-2": False}
        with self.assertRaises(SystemExit):
            Experiment._save_aggregated_params(aggregated_params_init=agg_params, breakpoint_path='/')

        # Test expected sceneries
        agg_params = {
            "node-1": {'params_path': '/'},
            "node-2": {'params_path': '/'}
        }
        expected_agg_params = {
            "node-1": {'params_path': '/test/path/'},
            "node-2": {'params_path': '/test/path/'}
        }
        agg_p = Experiment._save_aggregated_params(aggregated_params_init=agg_params, breakpoint_path='/')
        self.assertDictEqual(agg_p, expected_agg_params, '_save_aggregated_params result is not as expected')


    def test_experiment_31_static_load_aggregated_params(self):
        """ Testing static method for loading aggregated params of Experiment"""

        def load_func(x, to_params):
            return False

        # Test invalid type of aggregated params (should be dict)
        with self.assertRaises(SystemExit):
            Experiment._load_aggregated_params(True, load_func)

        # Test invalid type of load func params (should be callable)
        with self.assertRaises(SystemExit):
            Experiment._load_aggregated_params({}, True)

        # Test invalid key in aggregated params
        agg_params = {
            "node-1": {'params_path': '/test/path/'},
            "node-2": {'params_path': '/test/path/'}
        }
        with self.assertRaises(SystemExit):
            Experiment._load_aggregated_params(agg_params, load_func)

        # Test normal scenario
        agg_params = {
            "0": {'params_path': '/test/path/'},
            "1": {'params_path': '/test/path/'}
        }
        expected = {0: {'params_path': '/test/path/', 'params': False},
                    1: {'params_path': '/test/path/', 'params': False}}
        result = Experiment._load_aggregated_params(agg_params, load_func)
        self.assertDictEqual(result, expected, '_load_aggregated_params did not return as expected')


    def test_experiment_32_private_create_object(self):
        """tests `_create_object_ method :
        Importing class, creating and initializing multiple objects from
        breakpoint state for object and file containing class code
        """

        # Test if args is not instance of dict
        with self.assertRaises(SystemExit):
            Experiment._create_object(args=True)

        # Test if module does not exist
        args = {
            'class': 'Test',
            'module': 'test.test'
        }
        with self.assertRaises(SystemExit):
            Experiment._create_object(args=args)

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
            "   def load_state(self, state :str, **kwargs):\n" + \
            "       self._state = state\n"

        class_source_exception = \
            "class TestClassException:\n" + \
            "   def __init__(self, **kwargs):\n" + \
            "       self._kwargs = kwargs\n" + \
            "       raise Exception()\n" + \
            "   def load_state(self, state :str, **kwargs):\n" + \
            "       self._state = state\n"

        test_class_name = 'TestClass'
        module_name = 'testmodule'
        test_class_name_exc = 'TestClassException'
        module_name_exc = 'testmoduleexc'
        # arguments for creating object
        object_def = {
            'class': test_class_name,
            'module': module_name,
            'other': 'my_arbitrary_field'
        }

        # arguments for creating object for raising exception
        object_def_exception = {
            'class': test_class_name_exc,
            'module': module_name_exc,
            'other': 'my_arbitrary_field'
        }

        # optional object arguments
        object_kwargs = {'toto': 321, 'titi': 'dummy_par'}

        # input file contains code for object creation
        module_file_path = os.path.join(self.experimentation_folder_path, module_name + '.py')
        module_file_path_exc = os.path.join(self.experimentation_folder_path, module_name_exc + '.py')
        with open(module_file_path, "w") as f:
            f.write(class_source)
        with open(module_file_path_exc, "w") as f:
            f.write(class_source_exception)

        # action

        # Test `eval` exception while building class
        with patch.object(fedbiomed.researcher.experiment, 'eval') as m_eval:
            m_eval.side_effect = Exception
            with self.assertRaises(SystemExit):
                Experiment._create_object(object_def)

        # Test when class_code() (building class) raises error
        with self.assertRaises(SystemExit):
            Experiment._create_object(args=object_def_exception)

        # Test instantiate multiple objects of the class
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
