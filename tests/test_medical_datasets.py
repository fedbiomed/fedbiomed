import unittest
import os
import random
import shutil
import tempfile

from pathlib import Path, PosixPath
from random import randint, choice
from uuid import uuid4

import itk
import numpy as np
import pandas as pd
import torch

from monai.data import ITKReader
from monai.transforms import Compose, GaussianSmooth, Identity, LoadImage, PadListDataCollate, ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from fedbiomed.common.data import NIFTIFolderDataset
from fedbiomed.common.exceptions import FedbiomedDatasetError
from torchvision.transforms import Lambda
from monai.transforms import GaussianSmooth
from fedbiomed.common.data import MedicalFolderDataset, MedicalFolderBase


class TestNIFTIFolderDataset(unittest.TestCase):

    def setUp(self) -> None:
        # Create fake dataset

        self.n_classes = 3
        self.n_samples = [random.randint(2, 6) for _ in range(self.n_classes)]

        self.root = tempfile.mkdtemp()  # Creates and returns tempdir
        self._create_synthetic_dataset()


    def tearDown(self) -> None:
        shutil.rmtree(self.root)


    def test_nifti_folder_dataset_01_instantiation_correct(self):
        # correct instantiations
        NIFTIFolderDataset(self.root)
        NIFTIFolderDataset(self.root, None, None)
        NIFTIFolderDataset(self.root, transform=Identity(), target_transform=None)
        NIFTIFolderDataset(self.root, transform=None, target_transform=Identity())


    def test_nifti_folder_dataset_02_instantiation_incorrect(self):
        # incorrect instantiations

        # incorrect path - type or values
        for dir in (3, '~badaccount', '/not/existent/dir'):
            with self.assertRaises(FedbiomedDatasetError):
                NIFTIFolderDataset(dir)

        # empty path directory
        temp = tempfile.mkdtemp()
        with self.assertRaises(FedbiomedDatasetError):
            NIFTIFolderDataset(temp)
        # directory with no nifti file
        temp = tempfile.mkdtemp()
        tempsub = os.path.join(temp, 'subfolder')
        os.mkdir(tempsub)
        Path(os.path.join(tempsub, 'testfile')).touch()
        with self.assertRaises(FedbiomedDatasetError):
            NIFTIFolderDataset(temp)
        # directory unreadable
        temp = tempfile.mkdtemp()
        tempsub = os.path.join(temp, 'subfolder')
        os.mkdir(tempsub)
        os.chmod(tempsub, 0)
        with self.assertRaises(FedbiomedDatasetError):
            NIFTIFolderDataset(temp)

        def fonction():
            return True

        test_transform = fonction()

        # incorrectly typed transform functions
        with self.assertRaises(FedbiomedDatasetError):
            NIFTIFolderDataset(self.root, test_transform, None)
        with self.assertRaises(FedbiomedDatasetError):
            NIFTIFolderDataset(self.root, None, test_transform)


    def test_nifti_folder_dataset_03_indexation_correct(self):
        dataset = NIFTIFolderDataset(self.root)

        img, target = dataset[0]

        self.assertTrue(torch.is_tensor(img))
        self.assertEqual(img.dtype, torch.float32)

        self.assertTrue(isinstance(target, int))


    def test_nifti_folder_dataset_04_indexation_incorrect(self):
        dataset = NIFTIFolderDataset(self.root)

        # type error
        for index in ('toto', {}):
            with self.assertRaises(FedbiomedDatasetError):
                _ = dataset[index]
        # value error
        for index in (-2, len(dataset), len(dataset) + 10):
            with self.assertRaises(IndexError):
                _ = dataset[index]

        # transformation error (transform function do not match data)
        transformation = [
            [PadListDataCollate(), None],
            [None, PadListDataCollate()],
            [PadListDataCollate(), PadListDataCollate()]
        ]
        for transform, target_transform in transformation:
            dataset = NIFTIFolderDataset(self.root, transform, target_transform)
            with self.assertRaises(FedbiomedDatasetError):
                dataset[0]

        # unreadable sample file
        temp = tempfile.mkdtemp()
        tempsub = os.path.join(temp, 'subfolder')
        os.mkdir(tempsub)
        tempsubfile = os.path.join(tempsub, 'testfile.nii')
        Path(tempsubfile).touch()
        dataset = NIFTIFolderDataset(temp)
        os.chmod(tempsubfile, 0)
        with self.assertRaises(FedbiomedDatasetError):
            dataset[0]


    def test_nifti_folder_dataset_05_len(self):
        dataset = NIFTIFolderDataset(self.root)
        n_samples = len(dataset)

        self.assertEqual(n_samples, sum(self.n_samples))


    def test_nifti_folder_dataset_06_labels(self):
        dataset = NIFTIFolderDataset(self.root)

        # verify type of returned labels
        labels = dataset.labels()
        self.assertTrue(isinstance(labels, list))
        for label in labels:
            self.assertTrue(isinstance(label, str))

        # compare label list content
        self.assertEqual(sorted(labels), sorted(self.class_names))


    def test_nifti_folder_dataset_07_files(self):
        dataset = NIFTIFolderDataset(self.root)

        # verify type of returned files
        files = dataset.files()
        self.assertTrue(isinstance(files, list))
        for file in files:
            self.assertTrue(isinstance(file, Path))

        # compare label list content
        self.assertEqual(sorted([str(f) for f in files]), sorted([str(Path(f).expanduser().resolve()) for f in self.sample_paths]))


    def test_nifti_folder_dataset_08_getitem(self):

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

                file_index = self.sample_paths.index(Path(dataset.files()[index]))
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
    def test_nifti_folder_dataset_09_dataloader(self):
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

                self.sample_paths.append(Path(img_path).expanduser().resolve())
                self.sample_class.append(self.class_names.index(class_name))



def _create_synthetic_dataset(root, n_samples, tabular_file, index_col):
    # Image and target data
    fake_img_data = np.random.rand(10, 10, 10)
    img = itk.image_from_array(fake_img_data)

    # Generate subject ids
    subject_ids = [str(uuid4()) for _ in range(n_samples)]
    modalities = ['T1', 'T2', 'label']
    centers = [f'center_{uuid4()}' for _ in range(randint(3, 6))]

    demographics = pd.DataFrame()
    demographics.index.name = index_col

    for subject_id in subject_ids:
        subject_folder = os.path.join(root, subject_id)
        os.makedirs(subject_folder)

        # Create class folder
        for modality in modalities:
            modality_folder = os.path.join(subject_folder, modality)
            os.mkdir(modality_folder)
            img_path = os.path.join(modality_folder, f'image_{uuid4()}.nii.gz')
            itk.imwrite(img, img_path)

        # Add demographics information
        demographics.loc[subject_id, 'AGE'] = randint(15, 90)
        demographics.loc[subject_id, 'CENTER'] = choice(centers)
    demographics.to_csv(tabular_file)


def _create_wrong_formatted_folder_for_medical_folder(root, n_samples):

    subject_ids = [str(uuid4()) for _ in range(n_samples)]
    for subject_id in subject_ids:
        subject_folder = os.path.join(root, subject_id)
        os.makedirs(subject_folder)


class TestMedicalFolderDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.tabular_file = os.path.join(self.root, 'participants.csv')
        self.index_col = 'FOLDER_NAME'

        self.transform = {'T1': Lambda(lambda x: torch.flatten(x))}
        self.target_transform = {'label': GaussianSmooth()}

        self.n_samples = 10
        self.batch_size = 3

        print(f'Dataset folder located in: {self.root}')
        _create_synthetic_dataset(self.root, self.n_samples, self.tabular_file, self.index_col)

    def tearDown(self) -> None:
        if 'IXI' not in self.root:
            shutil.rmtree(self.root)

    def test_medical_folder_dataset_01_instantiating_dataset(self):
        dataset = MedicalFolderDataset(self.root, demographics_transform=lambda x: torch.Tensor([0.]))
        self._assert_batch_types_and_sizes(dataset)

        with self.assertRaises(FedbiomedDatasetError):
            dataset = MedicalFolderDataset(self.root, transform="Invalid")

        with self.assertRaises(FedbiomedDatasetError):
            dataset = MedicalFolderDataset(self.root, target_transform="Invalid")

    def test_medical_folder_dataset_02_cached_properties(self):
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col)
        print(dataset.demographics.head())
        print(dataset.demographics.head())

    def test_medical_folder_dataset_03_instantiation_with_demographics(self):
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col,
                                   demographics_transform = lambda x: torch.as_tensor(x['AGE']))
        self._assert_batch_types_and_sizes(dataset)

    def test_medical_folder_dataset_04_data_transforms(self):
        dataset = MedicalFolderDataset(self.root, transform=self.transform,
                                       demographics_transform=lambda x: torch.Tensor([0.]))
        (images, demographics), targets = dataset[0]
        self.assertTrue(images['T1'].dim() == 1)

    def test_medical_folder_dataset_05_target_transform(self):
        dataset = MedicalFolderDataset(self.root, target_transform=self.target_transform,
                                       demographics_transform=lambda x: torch.Tensor([0.]))
        (images, demographics), targets = dataset[0]
        self.assertEqual(images['T1'].shape, targets['label'].shape)

    def test_medical_folder_dataset_06_set_dataset_parameters(self):
        dataset = MedicalFolderDataset(self.root)

        with self.assertRaises(FedbiomedDatasetError):
            dataset.set_dataset_parameters("NONEDICTPARAMS")

        dataset.set_dataset_parameters({"tabular_file": self.tabular_file, "index_col": self.index_col})
        self.assertEqual(str(dataset.tabular_file), str(Path(self.tabular_file).expanduser().resolve()))
        self.assertEqual(dataset.index_col, self.index_col)

    def _assert_batch_types_and_sizes(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        (images, demographics), targets = iter(data_loader).next()

        self.assertIsInstance(images, dict)
        self.assertIsInstance(targets, dict)
        self.assertIsInstance(demographics, torch.Tensor)

        lengths = [len(b) for b in images.values()]
        lengths += [len(b) for b in targets.values()]
        lengths += [demographics.shape[0]]

        # Assert for batch size on modalities and demographics
        self.assertTrue(len(set(lengths)) == 1)


class TestMedicalFolderBase(unittest.TestCase):

    def setUp(self) -> None:

        self.root = tempfile.mkdtemp()
        self.tabular_file = os.path.join(self.root, 'participants.csv')
        self.index_col = 'FOLDER_NAME'

        self.transform = {'T1': Lambda(lambda x: torch.flatten(x))}
        self.target_transform = {'label': GaussianSmooth()}

        self.n_samples = 10
        self.batch_size = 3

        _create_synthetic_dataset(self.root, self.n_samples, self.tabular_file, self.index_col)

    def tearDown(self) -> None:

        if 'IXI' not in self.root:
            shutil.rmtree(self.root)
        pass

    def test_medical_folder_base_01_init(self):
        self.medical_folder_base = MedicalFolderBase()
        self.assertIsNone(self.medical_folder_base.root, "MedicalFolderBase root should not in empty initialization")

        self.medical_folder_base = MedicalFolderBase(root=self.root)
        self.assertIsInstance(self.medical_folder_base.root, PosixPath)
        self.assertEqual(str(self.medical_folder_base.root), self.root,
                         "MedicalFolderBase root should not in empty initialization")

        with self.assertRaises(FedbiomedDatasetError):
            self.medical_folder_base = MedicalFolderBase(root="unknown-folder-path")

        # Try to set root to None
        with self.assertRaises(FedbiomedDatasetError):
            self.medical_folder_base.root = None

        # If subjects has no modality folder
        dummy_root = tempfile.mkdtemp()
        _create_wrong_formatted_folder_for_medical_folder(dummy_root, 3)
        with self.assertRaises(FedbiomedDatasetError):
            self.medical_folder_base.root = dummy_root

        # Remove tmp folder
        shutil.rmtree(dummy_root)

        # If root has no subject folder
        dummy_root_2 = tempfile.mkdtemp()
        _create_wrong_formatted_folder_for_medical_folder(dummy_root, 0)
        with self.assertRaises(FedbiomedDatasetError):
            self.medical_folder_base.root = dummy_root_2

        # Remove tmp folder
        shutil.rmtree(dummy_root_2)

    def test_medical_folder_base_02_modalities(self):
        """Testing the method gets modalities from subject folder"""

        self.medical_folder_base = MedicalFolderBase(root=self.root)
        unique_modalities, all_modalities = self.medical_folder_base.modalities()

        self.assertIsInstance(all_modalities, list, "All modalities are not as expected")
        unique_modalities.sort()
        self.assertListEqual(unique_modalities, ["T1", "T2", "label"])

    def test_medical_folder_base_03_available_subjects(self):
        """Testing the method that extract available subjects for training"""
        self.medical_folder_base = MedicalFolderBase(root=self.root)
        file = self.medical_folder_base.read_demographics(self.tabular_file, self.index_col)
        complete_subject, missing_folders, missing_entries = \
            self.medical_folder_base.available_subjects(subjects_from_index=file.index)

        # Test results
        self.assertListEqual(missing_folders, [])
        self.assertListEqual(missing_entries, [])

    def test_medical_folder_base_04_read_demographics(self):
        self.medical_folder_base = MedicalFolderBase(root=self.root)

        with self.assertRaises(FedbiomedDatasetError):
            self.medical_folder_base.read_demographics(os.path.join(self.root, 'toto'), index_col=12)

        test_csv = pd.DataFrame([[1, 2, 3], [1, 2, 3]])
        test_csv.to_csv(os.path.join(self.root, 'toto.csv'))
        df = self.medical_folder_base.read_demographics(os.path.join(self.root, 'toto.csv'), index_col=1)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
