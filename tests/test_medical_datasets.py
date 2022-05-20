import unittest
import os
import tempfile
import random
import shutil
from pathlib import Path

import itk
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import ITKReader
from monai.transforms import LoadImage, ToTensor, Compose, Identity, PadListDataCollate

from fedbiomed.common.data import NIFTIFolderDataset
from fedbiomed.common.exceptions import FedbiomedDatasetError


class TestNIFTIFolderDataset(unittest.TestCase):
    def setUp(self) -> None:
        # Create fake dataset

        self.n_classes = 3
        self.n_samples = [random.randint(2, 6) for _ in range(self.n_classes)]

        self.root = tempfile.mkdtemp()  # Creates and returns tempdir
        self._create_synthetic_dataset()

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_instantiation_correct(self):
        # correct instantiations
        NIFTIFolderDataset(self.root)
        NIFTIFolderDataset(self.root, None, None)
        NIFTIFolderDataset(self.root, transform=Identity(), target_transform=None)
        NIFTIFolderDataset(self.root, transform=None, target_transform=Identity())

    def test_instantiation_incorrect(self):
        # incorrect instantiations

        # incorrect path type and values
        for dir in (3, '~badaccount', '/not/existent/dir'):
            with self.assertRaises(FedbiomedDatasetError):
                NIFTIFolderDataset(dir)

        # empty path directory
        with self.assertRaises(FedbiomedDatasetError):
            temp = tempfile.mkdtemp()
            NIFTIFolderDataset(temp)
        # directory with no nifti file
        with self.assertRaises(FedbiomedDatasetError):
            temp = tempfile.mkdtemp()
            tempsub = os.path.join(temp, 'subfolder')
            os.mkdir(tempsub)
            Path(os.path.join(tempsub, 'testfile')).touch()
            NIFTIFolderDataset(temp)        

        def fonction():
            pass
        # incorrect transform functions
        with self.assertRaises(FedbiomedDatasetError):
            _ = NIFTIFolderDataset(self.root, fonction, None)
        with self.assertRaises(FedbiomedDatasetError):
            _ = NIFTIFolderDataset(self.root, None, fonction)

    def test_indexation_correct(self):
        dataset = NIFTIFolderDataset(self.root)

        img, target = dataset[0]

        self.assertTrue(torch.is_tensor(img))
        self.assertEqual(img.dtype, torch.float32)

        self.assertTrue(isinstance(target, int))

    def test_indexation_incorrect(self):
        dataset = NIFTIFolderDataset(self.root)

        # type error
        for index in ('toto', {}):
            with self.assertRaises(FedbiomedDatasetError):
                _ = dataset[index]
        # value error
        for index in (-2, len(dataset), len(dataset) + 10):
            with self.assertRaises(IndexError):
                _ = dataset[index]

    def test_len(self):
        dataset = NIFTIFolderDataset(self.root)
        n_samples = len(dataset)

        self.assertEqual(n_samples, sum(self.n_samples))

    def test_labels(self):
        dataset = NIFTIFolderDataset(self.root)

        # verify type of returned labels
        labels = dataset.labels()
        self.assertTrue(isinstance(labels, list))
        for label in labels:
            self.assertTrue(isinstance(label, str))

        # compare label list content
        self.assertEqual(sorted(labels), sorted(self.class_names))

    def test_files(self):
        dataset = NIFTIFolderDataset(self.root)

        # verify type of returned files
        files = dataset.files()
        self.assertTrue(isinstance(files, list))
        for file in files:
            self.assertTrue(isinstance(file, Path))

        # compare label list content
        self.assertEqual(sorted([str(f) for f in files]), sorted([str(f) for f in self.sample_paths]))


    def test_getitem(self):

        # test all combination of using/not using a transformation
        # with identity transformation for the type of the data
        transformation = [
            [None, None],
            [Identity(), None],
            [None, Identity()],
            [Identity(), Identity()]
        ]

        for transform, target_transform in transformation:
            dataset = NIFTIFolderDataset(self.root, transform, target_transform)
            for index, [input, target] in enumerate(dataset):
                # test return types
                self.assertTrue(isinstance(input, torch.Tensor))
                self.assertTrue(isinstance(target, int))

                file_index = self.sample_paths.index(str(dataset.files()[index]))
                # check that targets match (need to compare label string as ordering may differ)
                self.assertEqual(dataset.labels()[target], self.class_names[self.sample_class[file_index]])

                # check that the input match (read content)
                # don't apply transformation as we're only doing idle transform
                synth_input_func = Compose([
                    LoadImage(ITKReader(), image_only=True),
                    ToTensor()
                ])
                synth_input = synth_input_func(self.sample_paths[file_index])
                self.assertTrue(torch.all(torch.eq(input, synth_input)))

            # check we read all the samples
            self.assertEqual(index + 1, sum(self.n_samples))

    # not really a unit test belonging to this class, but nice to have it => ok ?
    def test_dataloader(self):
        dataset = NIFTIFolderDataset(self.root)
        batch_size = len(dataset) // 2
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        img_batch, targets = iter(loader).next()

        self.assertEqual(len(targets), batch_size)
        self.assertEqual(len(img_batch), batch_size)

    def _create_synthetic_dataset(self):
        self.class_names = []
        self.sample_paths = []
        self.sample_class = []

        for class_i, n_samples in enumerate(self.n_samples):
            class_name = f'class_{class_i}'
            if class_name not in self.class_names:
                self.class_names.append(class_name)
            class_path = os.path.join(self.root, class_name)
            os.makedirs(class_path)

            # Create class folder
            for subject_i in range(n_samples):
                img_path = os.path.join(class_path, f'subject_{subject_i}.nii.gz')

                fake_img_data = np.random.rand(10, 10, 10)
                img = itk.image_from_array(fake_img_data)
                itk.imwrite(img, img_path)

                self.sample_paths.append(img_path)
                self.sample_class.append(self.class_names.index(class_name))


if __name__ == '__main__':
    unittest.main()
