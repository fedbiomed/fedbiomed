import unittest
import sys, io
from unittest.mock import MagicMock, patch
from pathlib import Path

import testsupport.mock_node_environ  # noqa (remove flake8 false warning)
from fedbiomed.node.environ import environ

import fedbiomed.node.cli
import fedbiomed.node.cli_utils
from fedbiomed.node.cli import add_database
from fedbiomed.common.data import DataLoadingPlan, MapperDP


class TestCli(unittest.TestCase):
    @staticmethod
    def mock_cli_input(x):
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
            MapperDP.__eq__ = test_mapper_eq

            fedbiomed.node.cli_utils.dataset_manager.add_database = MagicMock()
            add_database()

            dp = MapperDP()
            dp.map = {'T1': ['T1philips', 'T1siemens'],
                      'T2': ['T2'], 'label': ['label'],
                      'Tnon-exist': ['non-existing-modality']}

            fedbiomed.node.cli_utils.dataset_manager.add_database.assert_called_once_with(
                name='test-db-name',
                tags=['test-tag1', 'test-tag2'],
                data_type='medical-folder',
                description='',
                path='some/valid/path',
                dataset_parameters={
                    'tabular_file': 'some/valid/path',
                    'index_col': 0
                },
                data_loading_plan=DataLoadingPlan({'modalities_to_folders': dp})
            )

            dlp_arg = fedbiomed.node.cli_utils.dataset_manager.add_database.call_args[1]['data_loading_plan']
            self.assertIn('modalities_to_folders', dlp_arg)
            self.assertDictEqual(dlp_arg['modalities_to_folders'].map, dp.map)
            self.assertEqual(dlp_arg.name, 'test-dlp-name')


if __name__ == '__main__':
    unittest.main()
