import unittest
from unittest.mock import patch, MagicMock, create_autospec
import os

from torchvision.transforms import Compose as TorchCompose

from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedDatasetValueError, \
    FedbiomedLoadingBlockValueError, FedbiomedLoadingBlockError
from fedbiomed.common.data import FlambyDataset, FlambyLoadingBlockTypes, FlambyDatasetMetadataBlock, \
    DataLoadingPlan, discover_flamby_datasets
from testsupport.testing_data_loading_block import LoadingBlockForTesting, LoadingBlockTypesForTesting


class TestFlamby(unittest.TestCase):
    """
    Unit Tests for FLamby integration.
    """

    @patch("fedbiomed.common.data._flamby_dataset.discover_flamby_datasets",
           return_value={100: 'fed_flamby_test', **discover_flamby_datasets()})
    def test_flamby_01_loading_blocks(self, patched_discover):
        """Test that the custom DataLoadingBlocks for FlambyDataset work as expected."""
        # Base case for FlambyDatasetSelectorLoadingBlock when there are no errors
        dlb = FlambyDatasetMetadataBlock()
        dlb.metadata['flamby_dataset_name'] = 'fed_flamby_test'
        dlb.metadata['flamby_center_id'] = 42

        serialized_dataset_type = dlb.serialize()
        self.assertIn('flamby_dataset_name', serialized_dataset_type)
        self.assertIn('flamby_center_id', serialized_dataset_type)
        self.assertEqual(serialized_dataset_type['flamby_dataset_name'], 'fed_flamby_test')
        self.assertEqual(serialized_dataset_type['flamby_center_id'], 42)
        self.assertIn('loading_block_class', serialized_dataset_type)
        self.assertIn('loading_block_module', serialized_dataset_type)
        self.assertIn('dlb_id', serialized_dataset_type)
        _ = FlambyDatasetMetadataBlock().deserialize(serialized_dataset_type)  # assert no errors raised

        # Assert raises when dataset name is of wrong type
        serialized_dataset_type['flamby_dataset_name'] = 0
        with self.assertRaises(FedbiomedLoadingBlockValueError):
            _ = FlambyDatasetMetadataBlock().deserialize(serialized_dataset_type)
        # Assert raises when center id is of wrong type
        serialized_dataset_type['flamby_center_id'] = 'a string'
        with self.assertRaises(FedbiomedLoadingBlockValueError):
            _ = FlambyDatasetMetadataBlock().deserialize(serialized_dataset_type)
        # Assert raises when dataset name is not one of the flamby datasets
        serialized_dataset_type['flamby_dataset_name'] = 'non-existing name'
        with self.assertRaises(FedbiomedLoadingBlockValueError):
            _ = FlambyDatasetMetadataBlock().deserialize(serialized_dataset_type)
        # Assert calling apply gives correct result
        self.assertDictEqual(dlb.apply(),
                             {'flamby_dataset_name': 'fed_flamby_test',
                              'flamby_center_id': 42
                              })
        # Assert exception raised when calling apply on uninitialized loading block
        with self.assertRaises(FedbiomedLoadingBlockError):
            FlambyDatasetMetadataBlock().apply()

    def test_flamby_02_fed_class_initialization(self):
        """Test that initialization of the FedClass happens correctly.

        - test that FedClass is correctly initialized when we set the dlp
        """
        dataset = FlambyDataset()

        # Assert raises when dlp is not present
        with self.assertRaises(FedbiomedDatasetError):
            dataset._init_flamby_fed_class()
        with self.assertRaises(FedbiomedDatasetError):
            dataset.get_center_id()

        # define dlp
        dlb = FlambyDatasetMetadataBlock()
        dlb.metadata = {'flamby_dataset_name': 'fed_flamby_test',
                        'flamby_center_id': 0}
        dlp = DataLoadingPlan({FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA: dlb})

        # Assert base case where everything works as expected
        mocked_module = MagicMock()
        mocked_module.FedClass = MagicMock()
        with patch("fedbiomed.common.data._flamby_dataset.import_module", return_value=mocked_module):
            dataset.set_dlp(dlp)
            mocked_module.FedClass.assert_called_once_with(center=0, train=True, pooled=False)

        metadata = dataset.apply_dlb(None, FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA)  # assert no errors
        # Assert the returned metadata are correct
        self.assertEqual(metadata['flamby_dataset_name'], 'fed_flamby_test')
        self.assertEqual(metadata['flamby_center_id'], 0)
        self.assertEqual(dataset.get_center_id(), 0)

        # Assert raises when init is called twice
        with self.assertRaises(FedbiomedDatasetError):
            dataset._init_flamby_fed_class()

        # Clear and assert that FedClass was also cleared
        dataset.clear_dlp()
        self.assertIsNone(dataset.get_flamby_fed_class())
        self.assertIsNone(dataset.get_transform())
        with self.assertRaises(FedbiomedDatasetError):
            dataset.get_center_id()

        # Assert malformed dlps
        with self.assertRaises(FedbiomedDatasetError):
            dataset.set_dlp(DataLoadingPlan())
        with self.assertRaises(FedbiomedDatasetError):
            dataset.set_dlp(DataLoadingPlan({
                LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING: LoadingBlockForTesting()}))

        # Test ModuleNotFoundError correctly converted to FedbiomedDatasetError
        with patch("fedbiomed.common.data._flamby_dataset.import_module", side_effect=ModuleNotFoundError):
            with self.assertRaises(FedbiomedDatasetError):
                dataset.set_dlp(dlp)
            # Make sure that the FlambyDataset remained clean
            self.assertIsNone(dataset._dlp)
            self.assertIsNone(dataset.get_flamby_fed_class())
            self.assertIsNone(dataset.get_transform())

        # Assert FedbiomedDatasetError raised when something goes wrong while instantiating FedClass
        mocked_module = MagicMock()
        mocked_module.FedClass = MagicMock()
        mocked_module.FedClass.side_effect = FileNotFoundError  # random error that could realistically be raised
        with patch("fedbiomed.common.data._flamby_dataset.import_module", return_value=mocked_module):
            with self.assertRaises(FedbiomedDatasetError):
                dataset.set_dlp(dlp)
            # Make sure that the FlambyDataset remained clean
            self.assertIsNone(dataset._dlp)
            self.assertIsNone(dataset.get_flamby_fed_class())
            self.assertIsNone(dataset.get_transform())

    def test_flamby_03_transform(self):
        dataset = FlambyDataset()

        # Assert raises when argument is of incorrect type
        with patch("fedbiomed.common.data._flamby_dataset.isinstance", return_value=False):
            with self.assertRaises(FedbiomedDatasetValueError):
                dataset.init_transform('Wrong type')

        # define dlp
        dlb = FlambyDatasetMetadataBlock()
        dlb.metadata = {'flamby_dataset_name': 'fed_flamby_test',
                        'flamby_center_id': 0}
        dlp = DataLoadingPlan({FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA: dlb})

        transform = MagicMock(spec=TorchCompose)
        # Assert base case where everything works as expected
        dataset.init_transform(transform)

        def mock_init(transform, center, train, pooled):
            pass

        class MockedFlambyModule:
            def __init__(self):
                self.FedClass = create_autospec(mock_init)
        mocked_module = MockedFlambyModule()

        with patch("fedbiomed.common.data._flamby_dataset.import_module", return_value=mocked_module):
            dataset.set_dlp(dlp)
            mocked_module.FedClass.assert_called_once_with(transform=transform, center=0, train=True, pooled=False)

        self.assertEqual(transform, dataset.get_transform())

        # Assert raises when called after FedClass was already initialized
        with self.assertRaises(FedbiomedDatasetError):
            dataset.init_transform(transform)

        # Clear and assert transform is properly cleaned up
        dataset.clear_dlp()
        self.assertIsNone(dataset.get_transform())

    @unittest.skipUnless(bool(os.environ.get('DISCOVER_FLAMBY', False)),
                                        'Skipped because this requires manual download of the flamby datasets')
    def test_flamby_99_discover_flamby_datasets(self):
        """Test that all discovered datasets can be instantiated

        To run this test from command line, execute:
        ```
            DISCOVER_FLAMBY=True python test_flamby.py TestFlamby.test_flamby_04_discover_flamby_datasets
        ```

        We are forced to skip this test as it requires a manual download step for all datasets.
        However, it is still useful to retain it because it can be run manually to obtain a report on which datasets
        would work on the current Fed-BioMed and Flamby installations.
        """
        outcomes = {}  # {name(str) : outcome(str)}
        datasets = discover_flamby_datasets()

        dataset = FlambyDataset()
        for flamby_dataset_name in datasets.values():
            # define dlp
            dlb = FlambyDatasetMetadataBlock()
            dlb.metadata = {'flamby_dataset_name': flamby_dataset_name,
                            'flamby_center_id': 0}
            dlp = DataLoadingPlan({FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA: dlb})

            with patch("builtins.input", return_value="n"):
                try:
                    dataset.set_dlp(dlp)  # Assert that no errors are raised here
                except (FedbiomedDatasetError, SystemExit) as e:
                    print(f"{e}")
                    outcomes[flamby_dataset_name] = "FAIL"
                else:
                    outcomes[flamby_dataset_name] = "SUCCESS"
                dataset.clear_dlp()
        # Summary
        for name, outcome in outcomes.items():
            print(f"{name} : {outcome}")


if __name__ == "__main__":
    unittest.main()
