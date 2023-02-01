import os
import shutil
import unittest
from unittest.mock import patch, MagicMock


#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################


from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.monitor import Monitor, MetricStore  # noqa


def create_file(file_name: str):
    """ Creates a simple file for testing.

    Args:
        file_name (str): [path name of the file
    """
    # with tempfile.TemporaryFile() as temp_file:
    with open(file_name, "w") as f:
        f.write("this is a test. This file should be removed")


class TestMonitor(ResearcherTestCase):
    """
    Test `Monitor` class
    Args:
        unittest ([type]): [description]
    """

    # before the tests
    def setUp(self):

        self.patch_summary_writer = patch('torch.utils.tensorboard.SummaryWriter')
        self.patch_add_scalar = patch('fedbiomed.researcher.monitor.SummaryWriter.add_scalar')

        self.mock_summary_writer = self.patch_summary_writer.start()
        self.mock_add_scalar = self.patch_add_scalar.start()

        self.mock_summary_writer.return_value = MagicMock()
        self.mock_add_scalar.return_value = MagicMock()

        self.monitor = Monitor()

    # after the tests
    def tearDown(self):

        # Clean all files in tmp/runs tensorboard results directory
        for files in os.listdir(environ['TENSORBOARD_RESULTS_DIR']):
            path = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

        self.patch_summary_writer.stop()
        self.patch_add_scalar.stop()

        pass

    @patch('fedbiomed.researcher.monitor.Monitor._remove_logs')
    def test_monitor_01_initialization(self,
                                       patch_remove_logs):

        tensorboard_folder = self.monitor._log_dir
        self.assertTrue(os.path.isdir(tensorboard_folder))

        # create file inside
        test_file = os.path.join(tensorboard_folder, "test_file")
        create_file(test_file)

        # 2nd call to monitor
        _ = Monitor()
        patch_remove_logs.assert_called_once()

    def test_monitor_02_remove_logs_as_file(self):

        """ Test removing log files from directory
        using _remove_logs method
        """

        test_file = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], "test_file")
        create_file(test_file)
        self.monitor._remove_logs()
        self.assertFalse(os.path.isfile(test_file), "Tensorboard log file has not been removed properly")

    def test_monitor_03_remove_logs_if_directory(self):

        test_dir = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], 'test')
        os.mkdir(test_dir)
        test_file = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], 'test', "test_file")
        create_file(test_file)

        self.monitor._remove_logs()
        self.assertFalse(os.path.isfile(test_file), "Tensorboard log file has not been removed properly")
        self.assertFalse(os.path.isdir(test_dir), "Tensorboard log folder has not been removed properly")

    def test_monitor_05_summary_writer(self):
        """ Test writing loss value at the first step 0 iteration """

        node_id = "1234"
        self.monitor._summary_writer('Train', node_id, metric={"Loss": 12}, cum_iter=1)
        self.assertTrue('1234' in self.monitor._event_writers)
        self.mock_add_scalar.assert_called_once()

        self.mock_add_scalar.reset_mock()
        self.monitor._summary_writer('Train', node_id, metric={"Loss": 12, "Loss2": 14}, cum_iter=1)
        self.assertEqual(self.mock_add_scalar.call_count, 2)

    @patch('fedbiomed.researcher.monitor.Monitor._summary_writer')
    def test_monitor_06_on_message_handler(self, mock_summary_writer):

        """Test on_message_handler of Monitor class """

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(True)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'iteration': 1,
            'epoch': 1,
            'num_samples_trained': 13,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_called_once_with(header='VALIDATION ON GLOBAL UPDATES',
                                                    node='asd123',
                                                    metric={'metric_1': 12, 'metric_2': 13},
                                                    cum_iter=1)

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(True)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': True,
            'test': False,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'num_samples_trained': 13,
            'iteration': 1,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_called_once_with(header='TRAINING',
                                                    node='asd123',
                                                    metric={'metric_1': 12, 'metric_2': 13},
                                                    cum_iter=1)

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(False)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'num_samples_trained': 13,
            'iteration': 2,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_not_called()

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(True)
        self.monitor.set_tensorboard("not_a_bool")  # as a side effect, this will set tensorboard flag to false
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'num_samples_trained': 13,
            'iteration': 2,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_not_called()

    @patch('fedbiomed.researcher.monitor.SummaryWriter.close')
    def test_monitor_07_close_writers(self, mock_close):
        """  Testing closing writers """

        self.monitor._summary_writer(header='VALIDATION ON GLOBAL PARAMETERS',
                                     node='asd123',
                                     metric={'metric_1': 12, 'metric_2': 13},
                                     cum_iter=1)
        self.monitor.close_writer()
        mock_close.assert_called_once()


class TestMetricStore(unittest.TestCase):

    # before the tests
    def setUp(self):
        self.metric_store = MetricStore()

    # after the tests
    def tearDown(self):
        pass

    def test_metric_store_01_add_iteration_for_train_loss(self):
        """Testing add iteration method of the class MetricStore"""

        node_1 = 'node-1'
        node_2 = 'node-2'

        # Test first round for node 1 ----------------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected = {'node-1': {'testing_global_updates': {},
                               'testing_local_updates': {},
                               'training': {
                                   'Metric_1': {
                                       1: {'iterations': [1], 'values': [12]}}}
                               }}
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-1')
        self.assertEqual(cum_iter, 1)

        # Test first round for node 2 ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected.update({'node-2': {'testing_global_updates': {},
                                    'testing_local_updates': {},
                                    'training': {
                                        'Metric_1': {
                                            1: {'iterations': [1], 'values': [12]}}}
                                    }})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-2')
        self.assertEqual(cum_iter, 1)

        # Test first round for node 1 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-1']['training'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-1')
        self.assertEqual(cum_iter, 1)

        # Test first round for node 2 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-2']['training'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-2')
        self.assertEqual(cum_iter, 1)

        # Test second round for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-1']['training']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 2)

        # Test second round for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-2']['training']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-2')
        self.assertEqual(cum_iter, 2)

        # Test second round - second iter for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)
        expected['node-1']['training']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

        # Test second round - second iter for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=True,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)
        expected['node-2']['training']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

    def test_metric_store_02_add_iteration_for_test_on_global_updates(self):
        """Testing add iteration method of the class MetricStore"""

        node_1 = 'node-1'
        node_2 = 'node-2'

        # Test first round for node 1 ----------------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected = {'node-1': {'training': {},
                               'testing_local_updates': {},
                               'testing_global_updates': {
                                   'Metric_1': {
                                       1: {'iterations': [1], 'values': [12]}}}
                               }}
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-1')
        self.assertEqual(cum_iter, 1)

        # Test first round for node 2 ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected.update({'node-2': {'training': {},
                                    'testing_local_updates': {},
                                    'testing_global_updates': {
                                        'Metric_1': {
                                            1: {'iterations': [1], 'values': [12]}}}
                                    }})
        self.assertEqual(cum_iter, 1)

        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-2')

        # Test first round for node 1 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-1']['testing_global_updates'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-1')

        # Test first round for node 2 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-2']['testing_global_updates'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-2')

        # Test second round for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-1']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 2)

        # Test second round for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-2']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-2')
        self.assertEqual(cum_iter, 2)

        # Test second round - second iter for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)

        expected['node-1']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

        # Test second round - second iter for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)
        expected['node-2']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

        # Special for Test On Global Updates -------------------------------------------------------------------------

        # Test second round - second iter 2 again for node 1, it should overwrite -------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)

        expected['node-1']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second iteration could not set properly for the second '
                                                          'round of training for node-1')
        # Since it will overwrite on 2 iteration cum iter should 2
        self.assertEqual(cum_iter, 2)

        # Test second round - second iter 2 again for node 2, it should overwrite -------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=True,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-2']['testing_global_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second Iteration could not set properly for the second '
                                                          'round of training for node-1')
        # Since it will overwrite on 2 iteration cum iter should 2
        self.assertEqual(cum_iter, 2)

    def test_metric_store_03_add_iteration_for_test_on_local_updates(self):
        """Testing add iteration method of the class MetricStore"""

        node_1 = 'node-1'
        node_2 = 'node-2'

        # Test first round for node 1 ----------------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected = {'node-1': {'training': {},
                               'testing_global_updates': {},
                               'testing_local_updates': {
                                   'Metric_1': {
                                       1: {'iterations': [1], 'values': [12]}}}
                               }}
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-1')
        self.assertEqual(cum_iter, 1)

        # Test first round for node 2 ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected.update({'node-2': {'training': {},
                                    'testing_global_updates': {},
                                    'testing_local_updates': {
                                        'Metric_1': {
                                            1: {'iterations': [1], 'values': [12]}}}
                                    }})
        self.assertEqual(cum_iter, 1)

        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration of node-2')

        # Test first round for node 1 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-1']['testing_local_updates'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-1')

        # Test first round for node 2 for another metric ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=1,
                                                       metric={'Metric_2': 12},
                                                       iter_=1)
        expected['node-2']['testing_local_updates'].update({'Metric_2': {1: {'iterations': [1], 'values': [12]}}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the first '
                                                          'train iteration for another metric of node-2')

        # Test second round for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-1']['testing_local_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 2)

        # Test second round for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=1)
        expected['node-2']['testing_local_updates']['Metric_1'].update({2: {'iterations': [1], 'values': [12]}})
        self.assertDictEqual(self.metric_store, expected, 'Iteration could not set properly for the second '
                                                          'round of training for node-2')
        self.assertEqual(cum_iter, 2)

        # Test second round - second iter for node 1  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_1,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)
        expected['node-1']['testing_local_updates']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

        # Test second round - second iter for node 2  ---------------------------------------------------------
        cum_iter, *_ = self.metric_store.add_iteration(node=node_2,
                                                       train=False,
                                                       test_on_global_updates=False,
                                                       round_=2,
                                                       metric={'Metric_1': 12},
                                                       iter_=2)
        expected['node-2']['testing_local_updates']['Metric_1'].update({2: {'iterations': [1, 2], 'values': [12, 12]}})
        self.assertDictEqual(self.metric_store, expected, 'Second Iteration could not set properly for the second '
                                                          'round of training for node-1')
        self.assertEqual(cum_iter, 3)

    def test_metric_store_04_register_node(self):
        """Testing node registration for MetricStore """

        expected = {}
        node_1 = 'node-1'
        node_2 = 'node-2'
        self.metric_store._register_node(node_1)
        expected[node_1] = {
            "training": {},
            "testing_global_updates": {},  # Testing before training
            "testing_local_updates": {}  # Testing after training
        }
        self.assertDictEqual(self.metric_store, expected)

        self.metric_store._register_node(node_2)
        expected[node_2] = {
            "training": {},
            "testing_global_updates": {},  # Testing before training
            "testing_local_updates": {}  # Testing after training
        }
        self.assertDictEqual(self.metric_store, expected)

    def test_metric_store_05_register_metric(self):
        """Testing metric registration for MetricStore """

        expected = {'node-1': {
                        "training": {},
                        "testing_global_updates": {},  # Testing before training
                        "testing_local_updates": {}  # Testing after training
                    },
                    'node-2': {
                        "training": {},
                        "testing_global_updates": {},  # Testing before training
                        "testing_local_updates": {}  # Testing after training
                    }}

        self.metric_store._register_node('node-1')
        self.metric_store._register_node('node-2')

        # Register metric for node-1
        self.metric_store._register_metric('node-1', 'training', 'Metric_1')
        expected['node-1']['training'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

        # Register metric for node 2
        self.metric_store._register_metric('node-2', 'training', 'Metric_1')
        expected['node-2']['training'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

        # Register metric for node 1 - testing_global_updates
        self.metric_store._register_metric('node-1', 'testing_global_updates', 'Metric_1')
        expected['node-1']['testing_global_updates'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

        # Register metric for node 2 - testing_global_updates
        self.metric_store._register_metric('node-2', 'testing_global_updates', 'Metric_1')
        expected['node-2']['testing_global_updates'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

        # Register metric for node 1 - testing_local_updates
        self.metric_store._register_metric('node-1', 'testing_local_updates', 'Metric_1')
        expected['node-1']['testing_local_updates'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

        # Register metric for node 2 - testing_local_updates
        self.metric_store._register_metric('node-2', 'testing_local_updates', 'Metric_1')
        expected['node-2']['testing_local_updates'].update({'Metric_1': {1: {'iterations': [], 'values': []}}})
        self.assertDictEqual(self.metric_store, expected)

    def test_metric_store_06_cumulative_iteration(self):
        """Testing cumulative iteration calculation """

        rounds = {
            0: {'iterations': [1, 2, 3, 4], 'values': [10, 20, 30, 40]},
            1: {'iterations': [1, 2, 3, 4], 'values': [10, 20, 30, 40]}
        }

        cum_iter = self.metric_store._cumulative_iteration(rounds)
        self.assertEqual(cum_iter, 8)

        rounds = {
            0: {'iterations': [1, 2, 3, 4], 'values': [10, 20, 30, 40]},
            1: {'iterations': [1, 2], 'values': [10, 20]}
        }

        cum_iter = self.metric_store._cumulative_iteration(rounds)
        self.assertEqual(cum_iter, 6)

        rounds = {
            0: {'iterations': [1, 2, 3, 4], 'values': [10, 20, 30, 40]},
            1: {'iterations': [1], 'values': [10, 20]}
        }

        cum_iter = self.metric_store._cumulative_iteration(rounds)
        self.assertEqual(cum_iter, 5)

        rounds = {
            0: {'iterations': [1, 2, 3, 4], 'values': [10, 20, 30, 40]},
            1: {'iterations': [], 'values': []}
        }

        cum_iter = self.metric_store._cumulative_iteration(rounds)
        self.assertEqual(cum_iter, 4)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
