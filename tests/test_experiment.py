# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ

# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for researcher, since tests will be running for researcher component
from fedbiomed.researcher.environ import environ

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import sys
import shutil
import json
import pathvalidate
import inspect
import fedbiomed.researcher.experiment

from fedbiomed.common.torchnn import TorchTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.experiment import Experiment, exp_exceptions
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedSilentTerminationError
from tests.testsupport.fake_dataset import FederatedDataSetMock
from tests.testsupport.fake_experiment import ExperimentMock
from tests.testsupport.fake_training_plan import FakeModel


class TestExperiment(unittest.TestCase):
    class ZMQInteractiveShell:
        def __call__(self):
            pass

    # For testing model_class setter of Experiment
    class FakeModelTorch(TorchTrainingPlan):
        pass

    @classmethod
    def create_fake_model_file(cls, name: str):
        """ Class method saving codes of FakeModel

        Args:
            name (str): Name of the model file that will be created
        """

        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        tmp_dir_model = os.path.join(tmp_dir, name)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        content = inspect.getsource(FakeModel)
        file = open(tmp_dir_model, "w")
        file.write(content)
        file.close()

        return tmp_dir_model

    @classmethod
    def setUpClass(cls) -> None:

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

        # build minimal objects, needed to extract state by calling object method

        self.patchers = [
            patch('fedbiomed.researcher.datasets.FederatedDataSet',
                  FederatedDataSetMock),
            patch('fedbiomed.researcher.job.Job.__init__',
                  return_value=None),
            patch('fedbiomed.researcher.monitor.Monitor.__init__',
                  return_value=None),
            patch('fedbiomed.researcher.monitor.Monitor.on_message_handler',
                  return_value=False),
            patch('fedbiomed.researcher.requests.Requests.add_monitor_callback',
                  return_value=None),
            patch('fedbiomed.researcher.aggregators.fedavg.FedAverage.__init__',
                  return_value=None),
            patch('fedbiomed.researcher.aggregators.aggregator.Aggregator.__init__',
                  return_value=None)
        ]

        # Define patchers ---------------------------------------------------------------------------------------
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

        # Define mocks from patchers ---------------------------------------------------------------------------
        self.mock_create_folder = self.patcher_cr_folder.start()
        self.mock_logger_info = self.patcher_logger_info.start()
        self.mock_logger_error = self.patcher_logger_error.start()
        self.mock_logger_critical = self.patcher_logger_critical.start()
        self.mock_logger_debug = self.patcher_logger_debug.start()
        self.mock_logger_warning = self.patcher_logger_warning.start()
        self.mock_job = self.patcher_job.start()
        self.mock_request_init = self.patcher_request_init.start()
        self.mock_request_search = self.patcher_request_search.start()

        for patcher in self.patchers:
            patcher.start()

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

        # Test getter for model_class
        model_class = self.test_exp.model_class()
        self.assertIsNone(model_class, 'Getter for model_class did not return expected model_class')

        # Test getter for model_path
        model_path = self.test_exp.model_path()
        self.assertIsNone(model_path, 'Getter for model_path did not return expected model_path')

        # Test getter for model arguments
        model_args = self.test_exp.model_args()
        self.assertDictEqual(model_args, {}, 'Getter for model_args did not return expected value')

        # Test getter for training arguments
        training_args = self.test_exp.training_args()
        self.assertDictEqual(training_args, {}, 'Getter for model_class did not return expected value')

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

        # Test getter training_replies -----------------------------------------------------------------------------

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

        # Test getter for model instance---------------------------------------------------------------------------

        # Test when ._job is None
        self.test_exp._job = None
        self.mock_logger_error.reset_mock()
        model_instance = self.test_exp.model_instance()
        self.assertIsNone(model_instance, 'Getter for .model_instance did not return expected value None')
        self.mock_logger_error.assert_called_once()

        # Test when ._job is not None
        fake_model_instance = FakeModel()
        type(self.mock_job).model = PropertyMock(return_value=fake_model_instance)
        self.test_exp._job = self.mock_job
        model_instance = self.test_exp.model_instance()
        self.assertEqual(model_instance, fake_model_instance, 'Getter for model_instance did not return expected Model')

    @patch('builtins.print')
    def test_experiment_02_info(self, mock_print):
        """Testing the method .info() of experiment class """
        mock_print.return_value(None)
        self.test_exp.info()
        self.assertEqual(mock_print.call_count, 2, 'Printing info called unexpected times')

        # Test info by completing missing parts for proper .run
        mock_print.reset_mock()
        self.test_exp._fds = FederatedDataSetMock({'node-1': []})
        self.test_exp._job = self.mock_job
        self.test_exp._model_is_defined = True
        self.test_exp.info()
        self.assertEqual(mock_print.call_count, 2, 'Printing info called unexpected times')

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
        with patch.object(fedbiomed.researcher.experiment, 'get_ipython',
                          create=True) as m:
            m.side_effect = TestExperiment.ZMQInteractiveShell

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
        self.assertIsInstance(training_data, FederatedDataSet, 'Setter for training_data did not set '
                                                               'proper FederatedDataset object')

        # Test by passing training data as Node when the tags is None
        td_expected = None
        tags = self.test_exp.set_tags(tags=None)
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data, td_expected, 'Setter for training data is not set as expected: None')

        # Test by passing FederatedDataset object
        # Do not use mock otherwise it will raise a type error
        td_expected = FederatedDataSet({'node-1': [{'dataset_id': 'ids'}]})
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data, td_expected, 'Setter for training data did not set given FederatedDataset '
                                                     'object')

        # Test by passing dict
        td_expected = {'node-1': [{'dataset_id': 'ids'}]}
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data.data(), td_expected, 'Setter for training data did not set given '
                                                            'FederatedDataset object')

        # Test by passing invalid type of training_data
        td_expected = 12
        with self.assertRaises(SystemExit):
            training_data = self.test_exp.set_training_data(training_data=td_expected)

        # Test when job is not None
        self.mock_logger_debug.reset_mock()
        td_expected = {'node-1': [{'dataset_id': 'ids'}]}
        self.test_exp._job = MagicMock()
        training_data = self.test_exp.set_training_data(training_data=td_expected)
        self.assertEqual(training_data.data(), td_expected, 'Setter for training data did not set given '
                                                            'FederatedDataset object')
        self.assertEqual(self.mock_logger_debug.call_count, 2, "Logger debug is called unexpected times")

    def test_experiment_05_set_aggregator(self):
        """Testing setter for aggregator attribute of Experiment class"""

        # Set aggregator with None
        aggregator = self.test_exp.set_aggregator(aggregator=None)
        self.assertIsInstance(aggregator, FedAverage, 'Setter for aggregator did not set proper FedAverage instance')

        # Set aggregator with an built object
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
        self.test_exp.set_training_data(None)

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

        # Test when self._round_limit is not defined
        del self.test_exp._round_limit
        self.test_exp._round_current = 2
        rl_expected = 1
        round_limit = self.test_exp.set_round_limit(round_limit=rl_expected)
        self.assertIsNone(round_limit, 'Setter for round limit did not set round_limit to None, this expected '
                                       'behaviour when self._round_limit is not defined')

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
        with patch.object(fedbiomed.researcher.experiment, 'sanitize_filename') as m:
            self.mock_logger_warning.reset_mock()
            self.test_exp.set_experimentation_folder('test')
            self.mock_logger_warning.assert_called_once()

        # Test debug message when job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_experimentation_folder('12')
        self.mock_logger_debug.assert_called_once()

    def test_experiment_10_set_model_class(self):

        # Test setting model_class to None
        model_class = self.test_exp.set_model_class(None)
        self.assertIsNone(model_class, 'Model class is not set as None')

        # Setting model_class as string
        mc_expected = 'TestModel'
        model_class = self.test_exp.set_model_class(mc_expected)
        self.assertEqual(model_class, mc_expected, 'Model class is not set properly while setting it in `str` type')

        # Setting AttributeError when model path is not defined
        del self.test_exp._model_path
        self.test_exp.set_model_class(mc_expected)
        self.assertEqual(self.test_exp._model_is_defined, False)

        # Back to normal
        self.test_exp._model_path = None

        # Test by passing class
        model_class = self.test_exp.set_model_class(TestExperiment.FakeModelTorch)
        self.assertEqual(model_class, TestExperiment.FakeModelTorch,
                         'Model class is not set properly while setting it as class')

        # Test by passing class which has no subclass of one of the training plan
        with self.assertRaises(SystemExit):
            self.test_exp.set_model_class(FakeModel)

        # Test by passing class when self._model_path is not defined
        del self.test_exp._model_path
        model_class = self.test_exp.set_model_class(TestExperiment.FakeModelTorch)
        self.assertEqual(self.test_exp._model_is_defined, False)

        # Back to normal
        self.test_exp._model_path = None

        #  Test passing incorrect python identifier
        with self.assertRaises(SystemExit):
            self.test_exp.set_model_class('Fake Model')

        # Test passing class built object
        with self.assertRaises(SystemExit):
            model_class = self.test_exp.set_model_class(TestExperiment.FakeModelTorch())

        # Test passing class invalid type
        with self.assertRaises(SystemExit):
            model_class = self.test_exp.set_model_class(12)

        # Test passing class invalid type
        with self.assertRaises(SystemExit):
            model_class = self.test_exp.set_model_class({})

        # Test if ._job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_model_class('FakeModel')
        # There will be two logger.debug call
        #  First    : Experiment is not fully configured since model_path is still None
        #  Second   : Update Job since model_class has changed
        self.assertEqual(self.mock_logger_debug.call_count, 2, 'Logger debug is called unexpected time while setting '
                                                               'model class')

    def test_experiment_11_set_model_path(self):
        """ Testing setter for model_path of experiment """

        # Test model_path is None
        model_path = self.test_exp.set_model_path(None)
        self.assertIsNone(model_path, 'Setter for model_path did not set model_path to None')

        # Test passing path for model_file
        fake_model_path = self.create_fake_model_file('fake_model.py')
        model_path = self.test_exp.set_model_path(fake_model_path)
        self.assertEqual(model_path, fake_model_path, 'Setter for model_path did not set model_path properly')

        # Test
        with patch.object(fedbiomed.researcher.experiment, 'sanitize_filepath') as m:
            m.return_value = 'test'
            with self.assertRaises(SystemExit):
                self.test_exp.set_model_path(fake_model_path)

        # Test invalid type of model_path argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_model_path(12)

        # Test when mode class is also set
        self.test_exp.set_model_class('FakeModel')
        self.test_exp.set_model_path(fake_model_path)
        self.assertEqual(self.test_exp._model_is_defined, True, '_model_is_defined returns False even model_class and '
                                                                'model_path is fully configured')
        # Test if `._job` is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        self.test_exp.set_model_path(fake_model_path)
        # There will be one debug call. If model_is_defined is False there might be two calls.
        # Since _model_is_defined has become True with previous test block there will be only one call
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
        """ Testing setter for training arguments of Experiment """

        # Test setting model_args as in invalid type
        with self.assertRaises(SystemExit):
            self.test_exp.set_training_args(None)

        # Test setting model_args properly with dict
        ma_expected = {'arg-1': 100}
        model_args = self.test_exp.set_training_args(ma_expected)
        self.assertDictEqual(ma_expected, model_args, 'Training arguments has not been set correctly by setter')

        # Test setting model_args while the ._job is not None
        self.mock_logger_debug.reset_mock()
        self.test_exp._job = MagicMock(return_value=True)
        model_args = self.test_exp.set_training_args(ma_expected)
        # There will be one debug call.
        self.assertDictEqual(ma_expected, model_args, 'Training arguments has not been set correctly by setter')
        self.mock_logger_debug.assert_called_once()

    @patch('fedbiomed.researcher.job.Job')
    @patch('fedbiomed.researcher.job.Job.__init__')
    def test_experiment_14_set_job(self, mock_job_init, mock_job):
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
        self.test_exp._model_is_defined = True
        self.test_exp._fds = None
        self.mock_logger_debug.reset_mock()
        self.test_exp.set_job()
        self.mock_logger_debug.assert_called_once()

        # Back to normal
        self.test_exp._fds = True  # Assign any value to not make it None

        # Test proper set job when everything is ready to create Job
        self.test_exp._model_is_defined = True
        self.test_exp.set_model_class = TestExperiment.FakeModelTorch
        job = self.test_exp.set_job()
        self.assertIsInstance(job, Job, 'Job has not been set properly')

    def test_experiment_15_set_save_breakpoints(self):
        """ Test setter for save_breakpoints attr of experiment class """

        # Test invalid type of argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_save_breakpoints(None)

        # test valid type of argument
        sb = self.test_exp.set_save_breakpoints(True)
        self.assertTrue(sb, 'save_breakpoint has not been set correctly')

    def test_experiment_16_set_monitor(self):
        """ Test setter for monitor attribute of experiment class"""

        # Type invalid type for tensorboard argument
        with self.assertRaises(SystemExit):
            self.test_exp.set_monitor('FALSE')

        # Test set_monitor to True
        # At this point monitor is already defined, since in tensorboard is passed as
        # True while build self.test_exp in setUp method. Then setting monitor again
        # should log info about monitor is already defined, and result should instance of Monitor class
        self.mock_logger_info.reset_mock()
        monitor = self.test_exp.set_monitor(True)
        self.mock_logger_info.assert_called_once()
        self.assertIsInstance(monitor, Monitor)

        # Test tensorboard False
        monitor = self.test_exp.set_monitor(False)
        self.assertIsNone(monitor, 'Monitor has not been deactivated')

        # Test tensorboard False even monitor is in active
        # monitor should be inactive because of previous call of set_monitor
        self.mock_logger_info.reset_mock()
        monitor = self.test_exp.set_monitor(False)
        # Should be one call of logger.info that says Monitor is already inactive
        self.mock_logger_info.assert_called_once()
        self.assertIsNone(monitor, 'Monitor is not inactivate')

    @patch('fedbiomed.researcher.experiment.Experiment.breakpoint')
    @patch('fedbiomed.researcher.aggregators.fedavg.FedAverage.aggregate')
    @patch('fedbiomed.researcher.strategies.default_strategy.DefaultStrategy.refine')
    @patch('fedbiomed.researcher.job.Job.training_replies', new_callable=PropertyMock)
    @patch('fedbiomed.researcher.job.Job.start_nodes_training_round')
    @patch('fedbiomed.researcher.job.Job.update_parameters')
    @patch('fedbiomed.researcher.job.Job.__init__')
    def test_experiment_17_run_once(self,
                                    mock_job_init,
                                    mock_job_updates_params,
                                    mock_job_training,
                                    mock_job_training_replies,
                                    mock_strategy_refine,
                                    mock_fedavg_aggregate,
                                    mock_experiment_breakpoint):
        """ Testing run_once method of Experiment class """

        mock_job_init.return_value = None
        mock_job_training.return_value = None
        mock_job_training_replies.return_value = {self.test_exp.round_current(): 'reply'}
        mock_strategy_refine.return_value = ({'param': 1}, [12.2])
        mock_fedavg_aggregate.return_value = None
        mock_job_updates_params.return_value = None
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
        self.test_exp.set_model_class(TestExperiment.FakeModelTorch)
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
                         'incrrease=True')

    @patch('fedbiomed.researcher.experiment.Experiment.run_once')
    def test_experiment_18_run(self, mock_exp_run_once):
        """ Testing run method of Experiment class """

        def run_once_side_effect(increase):
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

    @patch('builtins.open')
    @patch('fedbiomed.researcher.job.Job.__init__', return_value=None)
    @patch('fedbiomed.researcher.job.Job.model_file', new_callable=PropertyMock)
    def test_experiment_19_model_file(self,
                                      mock_model_file,
                                      mock_job_init,
                                      mock_open):
        """ Testing getter model_file of the experiment class """

        m_open = MagicMock()
        m_open.read = MagicMock(return_value=None)
        m_open.close.return_value = None

        mock_open.return_value = m_open
        mock_model_file.return_value = 'path/to/model'

        # Test if ._job is not defined
        with self.assertRaises(SystemExit):
            self.test_exp.model_file()

        # Test if display is not bool
        with self.assertRaises(SystemExit):
            self.test_exp.model_file(display='not-bool')

        # Test when display is false
        self.test_exp.set_model_class(TestExperiment.FakeModelTorch)
        self.test_exp.set_job()
        result = self.test_exp.model_file(display=False)
        self.assertEqual(result,
                         'path/to/model',
                         f'model_file() returned {result} where it should have returned `path/to/model`')

        # Test when display is true
        result = self.test_exp.model_file(display=True)
        self.assertEqual(result,
                         'path/to/model',
                         f'model_file() returned {result} where it should have returned `path/to/model`')

        # Test if `open()` raises OSError
        mock_open.side_effect = OSError
        with self.assertRaises(SystemExit):
            result = self.test_exp.model_file(display=True)

    @patch('fedbiomed.researcher.experiment.create_unique_file_link')
    @patch('fedbiomed.researcher.experiment.create_unique_link')
    @patch('fedbiomed.researcher.experiment.choose_bkpt_file')
    # testing _save_breakpoint + _save_aggregated_params
    # (not exactly a unit test, but probably more interesting)
    def test_save_breakpoint(
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
        training_data = {'node1': 'dataset1', 'node2': 'dataset2'}
        # we want to test with non null values
        training_args = {'trarg1': 'my_string', 'trarg2': 444, 'trarg3': True}
        self.test_exp._training_args = training_args
        model_args = {'modelarg1': 'value1', 'modelarg2': 234, 'modelarg3': False}
        self.test_exp._model_args = model_args
        model_file = '/path/to/my/model_file.py'
        model_class = 'MyOwnTrainingPlan'
        round_current = 2
        aggregator_state = {'aggparam1': 'param_value', 'aggparam2': 987, 'aggparam3': True}
        strategy_state = {'stratparam1': False, 'stratparam2': 'my_strategy', 'aggparam3': 0.45}
        job_state = {'jobparam1': {'sub1': 1, 'sub2': 'two'}, 'jobparam2': 'myjob_value'}

        # aggregated_params
        agg_params = {
            'entry1': {'params_path': '/dummy/path/to/aggparams/params_path.pt'},
            'entry2': {'params_path': '/yet/another/path/other_params_path.pt'}
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

        # could also do: self.test_exp._set_round_current(round_current)
        self.test_exp._round_current = round_current

        # build minimal Job object
        class Job():
            def save_state(self, breakpoint_path):
                return job_state

        self.test_exp._job = Job()
        # patch Job model_class / model_file
        self.test_exp._job.model_file = model_file
        self.test_exp._job.model_class = model_class

        # action
        self.test_exp.breakpoint()

        # verification
        final_model_path = os.path.join(
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

    @patch('fedbiomed.researcher.experiment.Experiment.model_instance')
    @patch('fedbiomed.researcher.experiment.Experiment._create_object')
    @patch('fedbiomed.researcher.experiment.find_breakpoint_path')
    # test load_breakpoint + _load_aggregated_params
    # cannot test Experiment constructor, need to fake it
    # (not exactly a unit test, but probably more interesting)
    def test_load_breakpoint(self,
                             patch_find_breakpoint_path,
                             patch_create_object,
                             patch_model_instance
                             ):
        """ test `load_breakpoint` :
            1. if breakpoint file is json loadable
            2. if experiment is correctly configured from breakpoint
        """

        # breakpoint values
        bkpt_file = 'file_4_breakpoint'

        training_data = {'train_node1': 'my_first_dataset', 2: 243}
        training_args = {1: 'my_first arg', 'training_arg2': 123.45}
        model_args = {'modarg1': True, 'modarg2': 7.12, 'modarg3': 'model_param_foo'}
        model_path = '/path/to/breakpoint_model_file.py'
        model_class = 'ThisIsTheTrainingPlan'
        round_current = 1
        experimentation_folder = 'My_experiment_folder_258'
        aggregator = {'aggreg1': False, 'aggreg2': 'dummy_agg_param', 18: 'agg_param18'}
        strategy = {'strat1': 'test_strat_param', 'strat2': 421, 3: 'strat_param3'}
        aggregated_params = {
            '1': {'params_path': '/path/to/my/params_path_1.pt'},
            2: {'params_path': '/path/to/my/params_path_2.pt'}
        }
        job = {1: 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999}

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
        final_training_data = {'train_node1': 'my_first_dataset', '2': 243}
        final_training_args = {'1': 'my_first arg', 'training_arg2': 123.45}
        final_aggregator = {'aggreg1': False, 'aggreg2': 'dummy_agg_param', '18': 'agg_param18'}
        final_strategy = {'strat1': 'test_strat_param', 'strat2': 421, '3': 'strat_param3'}
        final_job = {'1': 'job_param_dummy', 'jobpar2': False, 'jobpar3': 9.999}

        # patch functions for loading breakpoint
        patch_find_breakpoint_path.return_value = self.experimentation_folder_path, bkpt_file

        def side_create_object(args, **kwargs):
            return args

        patch_create_object.side_effect = side_create_object

        class FakeModelInstance:
            def load(self, aggreg, to_params):
                return model_params

        patch_model_instance.return_value = FakeModelInstance()

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

        # action
        loaded_exp = Experiment.load_breakpoint(self.experimentation_folder_path)

        for p in patches_experiment:
            p.stop()

        # verification
        self.assertTrue(isinstance(loaded_exp, Experiment))
        self.assertEqual(loaded_exp._tags, final_tags)
        self.assertEqual(loaded_exp._fds, final_training_data)
        self.assertEqual(loaded_exp._aggregator, final_aggregator)
        self.assertEqual(loaded_exp._node_selection_strategy, final_strategy)
        self.assertEqual(loaded_exp._round_current, round_current)
        self.assertEqual(loaded_exp._round_limit, self.round_limit)
        self.assertEqual(loaded_exp._experimentation_folder, final_experimentation_folder)
        self.assertEqual(loaded_exp._model_class, model_class)
        self.assertEqual(loaded_exp._model_path, model_path)
        self.assertEqual(loaded_exp._model_args, model_args)
        self.assertEqual(loaded_exp._training_args, final_training_args)
        self.assertEqual(loaded_exp._job._saved_state, final_job)
        self.assertEqual(loaded_exp._aggregated_params, final_aggregated_params)
        self.assertTrue(loaded_exp._save_breakpoints)
        self.assertFalse(loaded_exp._monitor)

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
            'module': module_name,
            'other': 'my_arbitrary_field'
        }
        # optional object arguments
        object_kwargs = {'toto': 321, 'titi': 'dummy_par'}

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
