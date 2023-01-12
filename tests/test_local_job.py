import shutil
import os
import inspect
import unittest


#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

from fedbiomed.researcher.environ import environ
from unittest.mock import patch, MagicMock, PropertyMock
from fedbiomed.researcher.job import localJob
from testsupport.fake_training_plan import FakeModel


class TestLocalJob(ResearcherTestCase):

    @classmethod
    def create_fake_model(cls, name: str):
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


    def setUp(self):

        mock_data_manager = MagicMock(return_value=None)
        mock_data_manager.load = MagicMock(return_value=None)
        mock_data_manager.split = MagicMock(return_value=(None, None))

        # Set MagicMock for Model Instance of Local Job
        self.model = MagicMock(return_value=None)
        self.model.save = MagicMock(return_value=None)
        self.model.save_code = MagicMock(return_value=None)
        self.model.load = MagicMock(return_value={'model_params': True})
        self.model.set_dataset_path = MagicMock(return_value=None)
        self.model.training_data.return_value = mock_data_manager
        self.model.type = MagicMock(return_value=None)
        self.model.training_routine = MagicMock(return_value=None)



        type(self.model).dependencies = PropertyMock(return_value=['from os import mkdir'])
        # Global Local Job Object
        self.local_job = localJob(training_plan_class=self.model)

    def tearDown(self) -> None:

        # Remove if there is dummy model file
        tmp_dir = os.path.join(environ['TMP_DIR'], 'tmp_models')
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    def test_local_job_01_initialization_with_model_path(self):
        """ Testing Job initialization by passing model file with model_path"""

        model_file_path = TestLocalJob.create_fake_model('dummy_model.py')

        # Rebuild local jon for testing __init__
        self.local_job = localJob(training_plan_path=model_file_path,
                                  training_plan_class='FakeModel')

        self.assertEqual(self.local_job.training_plan.__class__.__name__, FakeModel.__name__,
                         'Provided model and model instance of Job do not match, '
                         'while initializing Job with static model python file')

        # Testing model_path with unsupported python module name
        model_file_path = TestLocalJob.create_fake_model('dummy.model.py')
        with self.assertRaises(SystemExit):
            self.local_job = localJob(training_plan_path=model_file_path,
                                      training_plan_class='FakeModel')

    def test_local_job_02_initialization_with_model_instance(self):
        """ Testing Local Job initialization by passing training_plan_class as python instance -> `built class`"""

        # Rebuild local jon for init test
        self.local_job = localJob(training_plan_class=self.model)
        self.assertEqual(self.local_job.training_plan.__class__.__name__, self.model.__class__.__name__,
                         'Provided model and model instance of Job do not match, '
                         'while initializing Local Job with already built model class')

    def test_local_job_03_initialization_with_model_arguments(self):
        """Testing Local Job initialization with model_arguments"""

        # Testing Local Job with model arguments
        args = {'args': True}
        # Rebuild local jon for testing __init__
        self.local_job = localJob(training_plan_class=FakeModel, model_args=args)
        self.assertDictEqual(args, self.local_job._model_args, 'Model arguments is not set properly')
        self.assertEqual(self.local_job.training_plan.__class__.__name__, 'FakeModel',
                         'Provided model and model instance of Local Job do not match, ')

    def test_local_job_05_setters_and_getters(self):

        model = self.local_job.model
        self.assertEqual(model, self.local_job.model, 'Getter did not return proper model instance')

        tr_args = {'args': True}
        self.local_job.training_args = tr_args
        self.assertEqual(tr_args, self.local_job.training_args,
                         'Setter or getter did not properly set or get training arguments')

    @patch('fedbiomed.common.logger.logger.error')
    def test_local_job_06_start_training(self, mock_logger_error):
        """ Test Local Job start_training method """

        # Set training arguments
        tr_args = {'args': True}
        self.local_job.training_args = tr_args
        # Start training
        self.local_job.start_training()
        self.model.training_routine.assert_called_once()

        # Test failure during training
        self.model.training_routine.side_effect = Exception
        self.local_job.start_training()
        mock_logger_error.assert_called_once()

        # Test failure while saving model parameters
        mock_logger_error.reset_mock()
        self.model.training_routine.side_effect = None
        self.model.save.side_effect = Exception
        self.local_job.start_training()
        mock_logger_error.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
