import unittest
import os
import tempfile
import random
import shutil

import itk
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedbiomed.common.data import NIFTIFolderDataset
from fedbiomed.common.exceptions import FedbiomedDatasetError


class TestNIFTIFolderDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Create fake dataset

        self.n_classes = 3
        self.n_samples = [random.randint(2, 6) for _ in range(self.n_classes)]

        self.root = tempfile.mkdtemp()  # Creates and returns tempdir
        self._create_synthetic_dataset()

    def test_instantiation(self):
        _ = NIFTIFolderDataset(self.root)

    def test_indexation(self):
        dataset = NIFTIFolderDataset(self.root)

        img, target = dataset[0]

        self.assertTrue(torch.is_tensor(img))
        self.assertTrue(torch.is_tensor(target))

        self.assertEqual(img.dtype, torch.float32)
        self.assertTrue(target.dtype, torch.long)

    def test_len(self):
        dataset = NIFTIFolderDataset(self.root)
        n_samples = len(dataset)

        self.assertEqual(n_samples, sum(self.n_samples))

    def test_getitem(self):

        dataset = NIFTIFolderDataset(self.root)
        for index, [input, target] in enumerate(dataset):
            # test return types
            self.assertTrue(isinstance(input, torch.Tensor))
            self.assertTrue(isinstance(target, torch.Tensor))

            ## we should also test data is the same as synthetic dataset
            ## but cannot do that with the current class methods
            #
            ## tensors are the same, with no transforms/target_transforms
            #self.assertEqual(len(input), len(synt_input))
            #self.assertTrue(torch.all(torch.eq(input, synth_input)))
            #
            # same for target

        # check we read all the samples
        self.assertEqual(index + 1, sum(self.n_samples))

    def test_dataloader(self):
        dataset = NIFTIFolderDataset(self.root)
        batch_size = len(dataset) // 2
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        img_batch, targets = iter(loader).next()

        self.assertEqual(len(targets), batch_size)
        self.assertEqual(len(img_batch), batch_size)

    def test_empty_folder_raises_error(self):
        with self.assertRaises(FedbiomedDatasetError):
            temp = tempfile.mkdtemp()
            NIFTIFolderDataset(temp)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def _create_synthetic_dataset(self):
        fake_img_data = np.random.rand(10, 10, 10)
        img = itk.image_from_array(fake_img_data)

        for class_i, n_samples in enumerate(self.n_samples):
            class_path = os.path.join(self.root, f'class_{class_i}')
            os.makedirs(class_path)

            # Create class folder
            for subject_i in range(n_samples):
                img_path = os.path.join(class_path, f'subject_{subject_i}.nii.gz')
                itk.imwrite(img, img_path)


if __name__ == '__main__':
    unittest.main()
