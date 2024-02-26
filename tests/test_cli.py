import unittest
import argparse
import sys, io, os
from unittest.mock import MagicMock, patch
from pathlib import Path

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

import fedbiomed
import fedbiomed.node.cli_utils

from fedbiomed.node.cli import (
    NodeCLI, 
    NodeControl, 
    DatasetArgumentParser, 
    TrainingPlanArgumentParser,
    start_node
)
from fedbiomed.node.cli_utils._medical_folder_dataset import get_map_modalities2folders_from_cli, \
    add_medical_folder_dataset_from_cli
from fedbiomed.node.cli_utils import add_database
from fedbiomed.common.data import MapperBlock
from fedbiomed.common.exceptions import FedbiomedError
from test_medical_datasets import patch_modality_glob, patch_is_modality_dir

class TestTrainingPlanArgumentParser(NodeTestCase):

    """Test case for node cli dataset argument parse """
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.tp_arg_pars = TrainingPlanArgumentParser(self.subparsers)

    def test_01_training_plan_argument_parser_initialize(self):
        """Tests training plan parser intialization"""
        self.tp_arg_pars.initialize()
        self.assertTrue("training-plan" in self.subparsers.choices)

        tp_choices  = self.subparsers.choices["training-plan"]._subparsers._group_actions[0].choices

        self.assertTrue("approve" in tp_choices)  #noqa
        self.assertTrue("list" in tp_choices)  #noqa
        self.assertTrue("delete" in tp_choices) #noqa
        self.assertTrue("reject" in tp_choices) #noqa
        self.assertTrue("view" in tp_choices) #noqa
        self.assertTrue("update" in tp_choices) #noqa

    @patch.object(fedbiomed.node.cli,  "imp_cli_utils")
    def test_02_training_plan_argument_parser_execute(self, cli_utils):
        """Tests training plan arugment parser actions"""

        self.tp_arg_pars.initialize()

        self.tp_arg_pars.delete()
        cli_utils.return_value.delete_training_plan.assert_called_once()

        self.tp_arg_pars.list()
        cli_utils.return_value.tp_security_manager.list_training_plans.assert_called_once()

        self.tp_arg_pars.view()
        cli_utils.return_value.view_training_plan.assert_called_once()

        self.tp_arg_pars.register()
        cli_utils.return_value.register_training_plan.assert_called_once()

        self.tp_arg_pars.reject()
        cli_utils.return_value.reject_training_plan.assert_called_once()

        self.tp_arg_pars.update()
        cli_utils.return_value.update_training_plan.assert_called_once()


class TestDatasetArgumentParser(NodeTestCase):
    """Test case for node cli dataset argument parse """
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.dataset_arg_pars = DatasetArgumentParser(self.subparsers)

    def test_01_dataset_argument_parser_initialize(self):
        """Test initialization"""

        self.dataset_arg_pars.initialize()
        self.assertTrue("dataset" in self.subparsers.choices)
        dataset_choices = self.subparsers.choices["dataset"]._subparsers._group_actions[0].choices

        self.assertTrue("add" in dataset_choices)  #noqa
        self.assertTrue("list" in dataset_choices)  #noqa
        self.assertTrue("delete" in dataset_choices) #noqa


        self.assertTrue("--mnist" in  dataset_choices["add"]._option_string_actions)
        self.assertTrue("--file" in  dataset_choices["add"]._option_string_actions)

        self.assertTrue("--all" in  dataset_choices["delete"]._option_string_actions)
        self.assertTrue("--only-mnist" in  dataset_choices["delete"]._option_string_actions)

    @patch.object(fedbiomed.node.cli,  "imp_cli_utils")
    def test_02_dataset_argument_parser_add(self, cli_utils):

        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "add"])
        self.dataset_arg_pars.add(args)
        cli_utils.return_value.add_database.assert_called_once_with()
        cli_utils.return_value.add_database.reset_mock()

        args = self.parser.parse_args(["dataset", "add", "--mnist", "test"])
        self.dataset_arg_pars.add(args)
        cli_utils.return_value.add_database.assert_called_once_with(interactive=False, path="test")
        cli_utils.return_value.add_database.reset_mock()


    @patch.object(fedbiomed.node.cli,  "imp_cli_utils")
    def test_03_dataset_argument_parser_delete(self, cli_utils):

        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "delete"])
        self.dataset_arg_pars.delete(args)
        cli_utils.return_value.delete_database.assert_called_once_with()
        cli_utils.return_value.delete_database.reset_mock()

        args = self.parser.parse_args(["dataset", "delete", "--all"])
        self.dataset_arg_pars.delete(args)
        cli_utils.return_value.delete_all_database.assert_called_once_with()
        cli_utils.return_value.delete_all_database.reset_mock()

        args = self.parser.parse_args(["dataset", "delete", "--only-mnist"])
        self.dataset_arg_pars.delete(args)
        cli_utils.return_value.delete_database.assert_called_once_with(interactive=False)
        cli_utils.return_value.delete_database.reset_mock()


    @patch.object(fedbiomed.node.cli,  "imp_cli_utils")
    def test_04_dataset_argument_parser_list(self, cli_utils):

        self.dataset_arg_pars.initialize()
        args = self.parser.parse_args(["dataset", "list"])
        self.dataset_arg_pars.list(args)

        cli_utils.return_value.dataset_manager.list_my_data.return_value = []
        cli_utils.return_value.dataset_manager.list_my_data.assert_called_once_with(verbose=True)


class TestNodeControl(NodeTestCase):
    """Test case for node control parser"""
    def setUp(self):

        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = NodeControl(self.subparsers) 

    def test_01_node_control_initialize(self):
        """Tests intialize"""

        self.control.initialize()
        self.assertTrue("start" in self.subparsers.choices)
        self.assertTrue("--gpu-only" in self.subparsers.choices["start"]._option_string_actions)  #noqa
        self.assertTrue("--gpu-num" in self.subparsers.choices["start"]._option_string_actions)  #noqa
        self.assertTrue("--gpu" in self.subparsers.choices["start"]._option_string_actions) #noqa

    @patch("fedbiomed.node.cli.Process")
    def test_02_node_control_start(self, process):

        self.control.initialize()
        args = self.parser.parse_args(["start"])
        os.environ["FEDBIOMED_ACTIVE_NODE_ID"] = "test-node-id"

        self.control.start(args)
        process.assert_called_once()

        process.return_value.join.side_effect = KeyboardInterrupt
        process.return_value.is_alive.side_effect = [True, False]
        with self.assertRaises(SystemExit):
            self.control.start(args)


    @patch('fedbiomed.node.node.Node', autospec=True)
    @patch('fedbiomed.node.cli_utils', autospec=True)
    def test_03_node_control__start(self, cli_utils, mock_node):
        """Tests node start"""
        self.env["TRAINING_PLAN_APPROVAL"] = True
        self.env["ALLOW_DEFAULT_TRAINING_PLANS"] = True

        args= {"gpu": False}
        start_node(args)
        mock_node.return_value.task_manager.assert_called_once()
        mock_node.return_value.task_manager.reset_mock()

        args= {"gpu": False}
        self.env["TRAINING_PLAN_APPROVAL"] = False
        start_node(args)
        mock_node.return_value.task_manager.assert_called_once()

        with patch.object(fedbiomed.node.cli, "logger") as logger:
            mock_node.return_value.task_manager.side_effect = FedbiomedError
            start_node(args)
            logger.critical.assert_called_once()
            logger.critical.reset_mock()

            mock_node.return_value.task_manager.side_effect = Exception
            start_node(args)
            logger.critical.assert_called_once()




class TestNodeCLI(NodeTestCase):
    """Tests main NodeCLI"""

    def setUp(self) -> None:
        pass


    def test_01_node_cli_init(self):
        """Tests intialization"""
        self.node_cli = NodeCLI()
        self.node_cli.parse_args(["--config", "config_n1", 'dataset', 'list'])




class TestCli(NodeTestCase):
    @staticmethod
    def mock_cli_input(x):
        """Requires that each test defines TestCli.inputs as a list"""
        return TestCli.inputs.pop(0)

    @staticmethod
    def patch_modality_glob(x, y):
        # We are globbing all subject folders
        for f in [Path('T1philips'), Path('T2'), Path('label')] + \
                 [Path('T1siemens'), Path('T2'), Path('label')] + \
                 [Path('non-existing-modality')]:
            yield f

    def setUp(self) -> None:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_cli_01_add_database_medical_folder(self):
        database_inputs = ['test-db-name', 'test-tag1,test-tag2', '', 'test-dlp-name']
        medical_folder_inputs = ['y', '0', 'y', '1', '1', '2', '3', '0', 'Tnon-exist', 'y', '4']

        with patch('pathlib.Path.glob', new=TestCli.patch_modality_glob) as patched_glob, \
                patch('pathlib.Path.is_dir', return_value=True) as patched_dir, \
                patch('fedbiomed.common.data.MedicalFolderBase.demographics_column_names',
                      return_value=['col1', 'col2']) as patched_column_names, \
                patch('fedbiomed.common.data.MedicalFolderBase.validate_MedicalFolder_root_folder',
                      return_value=Path('some/valid/path')) as patched_validate_root, \
                patch('fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input',
                      return_value='some/valid/path') as patched_val_path_in, \
                patch('fedbiomed.node.cli_utils._io.input', return_value='5') as patched_cli_input, \
                patch('fedbiomed.node.cli_utils._database.input',
                      new=lambda x: database_inputs.pop(0)) as patched_db_input, \
                patch('fedbiomed.node.cli_utils._medical_folder_dataset.input',
                      new=lambda x: medical_folder_inputs.pop(0)) as patched_medical_folder_input:
            # Need to override equality test to enable assert_called_once_with to function properly
            def test_mapper_eq(self, other):
                return self.map == other.map

            MapperBlock.__eq__ = test_mapper_eq

            fedbiomed.node.cli_utils.dataset_manager.add_database = MagicMock()
            add_database()

            dlb = MapperBlock()
            dlb.map = {'T1': ['T1philips', 'T1siemens'],
                       'T2': ['T2'], 'label': ['label'],
                       'Tnon-exist': ['non-existing-modality']}

            # TODO : test with DLP, when CLI DLP supported by future version
            # fedbiomed.node.cli_utils.dataset_manager.add_database.assert_called_once_with(
            #    name='test-db-name',
            #    tags=['test-tag1', 'test-tag2'],
            #    data_type='medical-folder',
            #    description='',
            #    path='some/valid/path',
            #    dataset_parameters={
            #        'tabular_file': 'some/valid/path',
            #        'index_col': 0
            #    },
            #    data_loading_plan=DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})
            # )

            # TODO : test with DLP, when CLI DLP supported by future version
            # dlp_arg = fedbiomed.node.cli_utils.dataset_manager.add_database.call_args[1]['data_loading_plan']
            # self.assertIn(MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS, dlp_arg)
            # self.assertDictEqual(dlp_arg[MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS].map, dlb.map)
            # self.assertEqual(dlp_arg.name, 'test-dlp-name')


class TestMedicalFolderCliUtils(NodeTestCase):
    @staticmethod
    def mock_input(x):
        return TestMedicalFolderCliUtils.inputs.pop(0)

    @staticmethod
    def mock_validated_input(type):
        return 'some/valid/path'

    def setUp(self) -> None:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    @patch('fedbiomed.node.cli_utils._medical_folder_dataset.input', new=mock_input.__func__)
    def test_medical_folder_cli_utils_01_get_map_modalities2folders_from_cli(self):
        modality_folder_names = ['Should map to T1']
        # scenario 1: 'Should map to T1' <-> 'T1'. Assumes T1 is second in the list of modalities provided by default
        TestMedicalFolderCliUtils.inputs = ['1']
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {'T1': ['Should map to T1']})

        # scenario 2: wrong inputs first, then same as above
        TestMedicalFolderCliUtils.inputs = ['wrong', '5', '1']
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {'T1': ['Should map to T1']})

        # scenario 3: add new name
        modality_folder_names = ['Should map to T1', 'Should map to Tnew']
        TestMedicalFolderCliUtils.inputs = ['1', '0', 'Tnew', 'y', '4']
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {'T1': ['Should map to T1'], 'Tnew': ['Should map to Tnew']})

        # scenario 4: More complexity with some mistakes added in
        modality_folder_names = ['Should map to T1', 'Should map to Tnew', 'Should also map to T1', 'Should map to T2']
        TestMedicalFolderCliUtils.inputs = ['1', '0', 'Tmistake', 'n', 'Tnew', '', '4', '5', '1', '2']
        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
        self.assertDictEqual(dlb.map, {'T1': ['Should map to T1', 'Should also map to T1'],
                                       'Tnew': ['Should map to Tnew'],
                                       'T2': ['Should map to T2']})

    @patch('fedbiomed.node.cli_utils._medical_folder_dataset.input', new=mock_input.__func__)
    @patch('fedbiomed.node.cli_utils._medical_folder_dataset.validated_path_input', new=mock_validated_input.__func__)
    @patch('fedbiomed.common.data.MedicalFolderBase.validate_MedicalFolder_root_folder',
           return_value=Path('some/valid/path'))
    @patch('fedbiomed.common.data.MedicalFolderBase.demographics_column_names', return_value=['col1', 'col2'])
    @patch('pathlib.Path.glob', new=patch_modality_glob)
    @patch('pathlib.Path.is_dir', new=patch_is_modality_dir)
    def test_medical_folder_cli_utils_02_load_medical_folder_dataset_from_cli(self,
                                                                              patch_validated_path_input,
                                                                              patch_validate_root_folder):
        # Scenario 1:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user selects a demographics file
        #    - user wants to configure a data loading plan
        TestMedicalFolderCliUtils.inputs = ['y', '0', 'y', '1', '1', '2', '3', '0', 'Tnon-exist', 'y', '4']

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(True, None, None)
        self.assertEqual(path, 'some/valid/path')
        self.assertDictEqual(dataset_parameters, {
            'tabular_file': 'some/valid/path',
            'index_col': 0
        })
        # TODO : test with DLP, when CLI DLP supported by future version
        # self.assertIn(MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS, dlp)
        # self.assertDictEqual(dlp[MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS].map, {
        #    'T1': ['T1philips', 'T1siemens'],
        #    'T2': ['T2'],
        #    'Tnon-exist': ['non-existing-modality'],
        #    'label': ['label']
        # })

        # Scenario 2:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user selects a demographics file
        #    - user does not configure a data loading plan
        TestMedicalFolderCliUtils.inputs = ['y', '0', '']

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(True, None, None)
        self.assertEqual(path, 'some/valid/path')
        self.assertDictEqual(dataset_parameters, {
            'tabular_file': 'some/valid/path',
            'index_col': 0
        })
        self.assertIsNone(dlp)

        # Scenario 2:
        #    - no pre-existing dataset parameters or data loading plan
        #    - user does not select a demographics file
        #    - user does not configure a data loading plan
        TestMedicalFolderCliUtils.inputs = ['n', '']

        path, dataset_parameters, dlp = add_medical_folder_dataset_from_cli(True, None, None)
        self.assertEqual(path, 'some/valid/path')
        self.assertDictEqual(dataset_parameters, {})
        self.assertIsNone(dlp)


if __name__ == '__main__':
    unittest.main()
