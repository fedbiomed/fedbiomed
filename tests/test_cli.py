import unittest
import sys, io
from unittest.mock import MagicMock, patch
from pathlib import Path

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

import fedbiomed.node.cli_utils
from fedbiomed.node.cli_utils._medical_folder_dataset import get_map_modalities2folders_from_cli, \
    add_medical_folder_dataset_from_cli
from fedbiomed.node.cli_utils import add_database
from fedbiomed.common.data import MapperBlock
from test_medical_datasets import patch_modality_glob, patch_is_modality_dir


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
