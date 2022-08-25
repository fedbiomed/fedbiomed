import fedbiomed.node.flamby_utils as flutils
import fedbiomed.node.flamby_split as flsplit
import unittest
from torchvision.transforms import Compose as TorchCompose
from monai.transforms import Compose as MonaiCompose
from fedbiomed.common.training_plans import TorchTrainingPlan


class TestFlamby(unittest.TestCase):
    """
    Unit Tests for FLamby integration.
    """

    def setUp(self):
        """
        run this at the begining of each test
        """
        self.available_flamby_datasets = {1: 'flamby.datasets.fed_camelyon16',
                                          2: 'flamby.datasets.fed_dummy_dataset',
                                          3: 'flamby.datasets.fed_heart_disease',
                                          4: 'flamby.datasets.fed_isic2019',
                                          5: 'flamby.datasets.fed_ixi',
                                          6: 'flamby.datasets.fed_kits19',
                                          7: 'flamby.datasets.fed_lidc_idri',
                                          8: 'flamby.datasets.fed_synthetic',
                                          9: 'flamby.datasets.fed_tcga_brca'}
        self.valid_flamby_options = {1: 'camelyon16',
                                     2: 'dummy_dataset',
                                     3: 'heart_disease',
                                     4: 'isic2019',
                                     5: 'ixi',
                                     6: 'kits19',
                                     7: 'lidc_idri',
                                     8: 'synthetic',
                                     9: 'tcga_brca'}
        self.available_and_valid_flamby = (self.available_flamby_datasets, self.valid_flamby_options)

        self.fake_flamby_dataset = {"name": "ixi",
                                    "data_type": "flamby",
                                    "tags": "ixi",
                                    "description": "desc",
                                    "shape": [200, 1, 48, 60, 48],
                                    "path": "/",
                                    "dataset_id": "dataset_1234",
                                    "dataset_parameters": {"center_id": 0, "fed_class": "flamby.datasets.fed_ixi", "flamby": True},}

        self.testing_arguments = {}
        self.transform_compose = MonaiCompose([])

        self.model_kwargs = {
            'batch_size': 2,
            'lr': 0.001,
            'epochs': 1,
        }

        self.fake_model = TorchTrainingPlan(self.model_kwargs)


    def tearDown(self):
        """
        after each test function
        """
        pass

    def test_flamby_utils_01_get_flamby_datasets(self):
        """ Testing get_flamby_datasets function
        """
        res = flutils.get_flamby_datasets()
        self.assertTupleEqual(res, self.available_and_valid_flamby)

    def test_flamby_utils_02_get_transform_compose_flamby(self):
        """ Testing get_transform_compose_flamby function
        """
        res = flutils.get_transform_compose_flamby(["from monai.transforms import (Compose, NormalizeIntensity, Resize,)",
                                                    "Compose([Resize((48,60,48)), NormalizeIntensity()])"])
        self.assertIsInstance(res, MonaiCompose)

    def test_flamby_split_03_set_training_testing_data_loaders_flamby(self):
        """ Testing _set_training_testing_data_loaders_flamby function
        """
        res = flsplit._set_training_testing_data_loaders_flamby(
            self.fake_flamby_dataset, self.fake_model, self.testing_arguments, self.transform_compose, self.model_kwargs['batch_size'])
        self.assertIsInstance(res, TorchTrainingPlan)


if __name__ == "__main__":
    unittest.main()
