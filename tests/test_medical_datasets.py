import logging
import tempfile
import unittest

from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

from fedbiomed.common.data import NIFTIFolderDataset


class TestNIFTIFolderDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Set-up logger for debugging
        logging.basicConfig(level=logging.DEBUG)

        # Create fake dataset
        import random
        self.n_classes = 3
        self.n_samples = [random.randint(2, 6) for _ in range(self.n_classes)]

        self.root = tempfile.mkdtemp()  # Creates and returns tempdir
        log.debug(f'Dataset folder located in: {self.root}')
        self._create_synthetic_dataset()

    def test_instantiation(self):
        _ = NIFTIFolderDataset(self.root)

    def test_indexation(self):
        import torch
        dataset = NIFTIFolderDataset(self.root)
        logging.debug(dataset.files)

        img, target = dataset[0]

        self.assertTrue(torch.is_tensor(img))
        self.assertTrue(torch.is_tensor(target))

        self.assertEqual(img.dtype, torch.float32)
        self.assertTrue(target.dtype, torch.long)

    def test_len(self):
        dataset = NIFTIFolderDataset(self.root)
        n_samples = len(dataset)

        self.assertEqual(n_samples, sum(self.n_samples))
        self.assertEqual(len(dataset.targets), n_samples)
        self.assertEqual(len(dataset.files), n_samples)

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        dataset = NIFTIFolderDataset(self.root)
        batch_size = len(dataset) // 2
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        img_batch, targets = iter(loader).next()

        self.assertEqual(len(targets), batch_size)
        self.assertEqual(len(img_batch), batch_size)

    def test_empty_folder_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            temp = tempfile.mkdtemp()
            NIFTIFolderDataset(temp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.root)

    def _create_synthetic_dataset(self):
        import itk
        import os
        import numpy as np

        fake_img_data = np.random.rand(10, 10, 10)
        img = itk.image_from_array(fake_img_data)

        for class_i, n_samples in enumerate(self.n_samples):
            class_path = os.path.join(self.root, f'class_{class_i}')
            os.makedirs(class_path)

            # Create class folder
            for subject_i in range(n_samples):
                img_path = os.path.join(class_path, f'subject_{subject_i}.nii.gz')
                itk.imwrite(img, img_path)


import torch
from torchvision.transforms import Lambda
from monai.transforms import GaussianSmooth
from fedbiomed.common.data import BIDSDataset


class TestBIDSDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.root = '/Users/ssilvari/Downloads/IXI_sample'
        self.tabular_file = '/Users/ssilvari/Downloads/IXI_sample/participants.xlsx'
        self.index_col = 'FOLDER_NAME'

        self.transform = {'T1': Lambda(lambda x: torch.flatten(x))}
        self.target_transform = {'label': GaussianSmooth()}

        self.batch_size = 3

    def test_instantiating_dataset(self):
        dataset = BIDSDataset(self.root)
        batch = dataset[0]
        self.assertIsInstance(batch, dict)
        self.assertIsInstance(batch['data'], dict)
        self.assertIsInstance(batch['target'], dict)
        self.assertIsInstance(batch['demographics'], dict)

    def test_instantiation_with_demographics(self):
        dataset = BIDSDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col)
        batch = dataset[0]
        self.assertIsInstance(batch, dict)
        self.assertIsInstance(batch['data'], dict)
        self.assertIsInstance(batch['target'], dict)
        self.assertIsInstance(batch['demographics'], dict)

        # Use dataloader
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        batch = iter(data_loader).next()

        lengths = [len(b) for b in batch['data'].values()]
        lengths += [len(b) for b in batch['target'].values()]
        lengths += [len(b) for b in batch['demographics'].values()]

        # Assert for batch size on modalities and demographics
        self.assertTrue(len(set(lengths)) == 1)

    def test_data_transforms(self):
        dataset = BIDSDataset(self.root, transform=self.transform)
        batch = dataset[0]
        self.assertTrue(batch['data']['T1'].dim() == 1)

    def test_target_transform(self):
        dataset = BIDSDataset(self.root, target_transform=self.target_transform)
        batch = dataset[0]
        self.assertEqual(batch['data']['T1'].shape, batch['target']['label'].shape)


if __name__ == '__main__':
    unittest.main()
