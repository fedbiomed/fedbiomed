import io
import sys

import unittest
import os
import random
from random import randint, choice

import shutil
import tempfile
from pathlib import Path, PosixPath
from unittest.mock import patch, MagicMock
from uuid import uuid4

import itk
import monai
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from monai.data import ITKReader
from monai.transforms import LoadImage, ToTensor, Compose, Identity, PadListDataCollate, GaussianSmooth
from fedbiomed.common.data import NIFTIFolderDataset
from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedLoadingBlockError
from torch.utils.data import Dataset
from torchvision.transforms import Lambda
from fedbiomed.common.data import MedicalFolderDataset, MedicalFolderBase, MedicalFolderController,\
                                  MedicalFolderLoadingBlockTypes, DataLoadingPlan, MapperBlock


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
        # here we test that each instantation is a `NIFTIFolderDataset`
        # object, but the true goal of the test is to check that parameters
        # are accepted when initializing object

        self.assertIsInstance(NIFTIFolderDataset(self.root),
                              NIFTIFolderDataset)

        self.assertIsInstance(
            NIFTIFolderDataset(self.root, None, None),
            NIFTIFolderDataset)
        self.assertIsInstance(
            NIFTIFolderDataset(self.root, transform=Identity(),
                               target_transform=None),
            NIFTIFolderDataset)
        self.assertIsInstance(NIFTIFolderDataset(self.root, transform=None, target_transform=Identity()),
                              NIFTIFolderDataset)

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
        self.assertEqual(sorted([str(f) for f in files]),
                         sorted([str(Path(f).expanduser().resolve()) for f in self.sample_paths]))


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
        img_batch, targets = next(iter(loader))

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


def _create_synthetic_dataset(root: str, n_samples: int, tabular_file: str, index_col: str):
    """Creates synthetic dataset for test purposes

    Args:
        root (str): path to dataset
        n_samples (int): number of samples
        tabular_file (str): path to demographic dataset
        index_col (str): column name for subject id
    """
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


def _create_wrong_formatted_folder_for_medical_folder(root: str, n_samples: int):
    """Creates medical folder without any modalities

    Args:
        root (str): root path file
        n_samples (int): number of samples (ie number of subjects)
    """

    subject_ids = [str(uuid4()) for _ in range(n_samples)]
    for subject_id in subject_ids:
        subject_folder = os.path.join(root, subject_id)
        os.makedirs(subject_folder)

## Utilities for testin DataLoadingPlan
modalities_to_folders = {
    'T1': ['T1siemens', 'T1philips'],
    'T2': ['T2'],
    'label': ['label']
}
all_folder_names = [folder for folders in modalities_to_folders.values() for folder in folders]

def patch_is_modality_dir(x):
    """Mock the situation where:
        subj1 has philips but not siemens,
        subj2 has siemens but not philips
    """
    if x.name == 'T1siemens' and x.match('*/subj1/*'):
        return False
    elif x.name == 'T1philips' and x.match('*/subj2/*'):
        return False
    elif x.match('*/subj3/*'):
        return x.name == 'non-existing-modality'
    return True


def patch_modality_iterdir(x):
    if x.match('*/subj1'):
        return [Path('T1philips'), Path('T2'), Path('label')]
    if x.match('*/subj2'):
        return [Path('T1siemens'), Path('T2'), Path('label')]
    if x.match('*/subj3'):
        return [Path('non-existing-modality')]
    return [Path('subj1'), Path('subj2'), Path('subj3')]


def patch_modality_glob(self, x):
    if self.name not in all_folder_names:
        # We are globbing all subject folders
        for f in [Path('T1philips'), Path('T2'), Path('label')] + \
                 [Path('T1siemens'), Path('T2'), Path('label')] + \
                 [Path('non-existing-modality')]:
            yield f
    else:
        # We are globbing one specific modality folder
        yield Path(self.name + '_test.nii')


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
        dataset = MedicalFolderDataset(self.root)
        self._assert_batch_types_and_sizes(dataset)

        def dummy_transform(*args, **kwargs):
            return True

        for transform in "Invalid", \
                         dummy_transform, \
                         ["Invalid"], \
                         [dummy_transform], \
                         ["Invalid", "Invalid"], \
                         [dummy_transform, dummy_transform], \
                         {'T1': "Invalid"}, \
                         {'T3': dummy_transform }, \
                         {'T1': dummy_transform, 'T2': dummy_transform, 'T3': dummy_transform }, \
                         {'T1': dummy_transform, 'T2': "Invalid"}:

            with self.assertRaises(FedbiomedDatasetError):
                dataset = MedicalFolderDataset(self.root, data_modalities=['T1', 'T2'], transform=transform)

            with self.assertRaises(FedbiomedDatasetError):
                dataset = MedicalFolderDataset(self.root, target_modalities=['T1', 'T2'], target_transform=transform)

        for modalities in 'T1', ['T1']:
            for transform in "Invalid", \
                             ["Invalid"], \
                             [dummy_transform], \
                             ["Invalid", "Invalid"], \
                             [dummy_transform, dummy_transform], \
                             {'T1': "Invalid"}, \
                             {'T3': dummy_transform }, \
                             {'T1': dummy_transform, 'T3': dummy_transform }:

                with self.assertRaises(FedbiomedDatasetError):
                    dataset = MedicalFolderDataset(self.root, data_modalities=modalities, transform=transform)

                with self.assertRaises(FedbiomedDatasetError):
                    dataset = MedicalFolderDataset(self.root, target_modalities=modalities, target_transform=transform)



    def test_medical_folder_dataset_02_cached_properties(self):
        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col)
        self.assertIsInstance(dataset, Dataset)  # check that instantiation has been completed
        print(dataset.demographics.head())
        print(dataset.demographics.head())

    def test_medical_folder_dataset_03_getitem(self):
        # test correct indexation

        # test errors
        self.patcher = patch('monai.transforms.GaussianSmooth', side_effect=RuntimeError)
        self.patcher.start()
        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,
                                       transform= monai.transforms.GaussianSmooth,
                                       )
        with self.assertRaises(FedbiomedDatasetError):
            dataset[0]


        # test case where demographic transform raises error

        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,
                                       demographics_transform= monai.transforms.GaussianSmooth,
                                       )
        try:
            with self.assertRaises(FedbiomedDatasetError):
                dataset[0]
        finally:
            self.patcher.stop()  # make sure patcher is stopped (in order to avid propegating patch to other tests)

        # test case where `demographics` type is not correct

        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,
                                       demographics_transform= Lambda(
                                           lambda x: 'this is a bad type for demographics'
                                           ' ( expecting a dict but passing a str)'),
                                       )
        with self.assertRaises(FedbiomedDatasetError):
            dataset[0]

        self.patcher.start()

        dataset = MedicalFolderDataset(root=self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,
                                       target_transform={"label": monai.transforms.GaussianSmooth},
                                       demographics_transform=lambda x: torch.as_tensor(x['AGE'])
                                       )
        try:
            with self.assertRaises(FedbiomedDatasetError):
                dataset[0]
        finally:
            self.patcher.stop()   # make sure patcher is stopped (in order to avoid propagating patch to other tests)

    def test_medical_folder_dataset_04_len(self):
        dataset = MedicalFolderDataset(self.root)

        # check correct use of number of smaples
        self.assertEqual(len(dataset), self.n_samples)

        # check __len__ behaviour when self.subject_folder returns an empty list
        patcher = patch('fedbiomed.common.data._medical_datasets.MedicalFolderDataset.subject_folders',
                        return_value = [])
        patcher.start()
        dataset = MedicalFolderDataset(self.root)
        try:
            with self.assertRaises(FedbiomedDatasetError):
                len(dataset)
        finally:
            patcher.stop()  # make sure patcher is stopped (otherwise will propagate error)

    def test_medical_folder_dataset_05_tabular_data_setter(self):
        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,)

        # test with a temporary file

        tmp_file = tempfile.NamedTemporaryFile()
        dataset.tabular_file = tmp_file.name
        self.assertEqual(str(dataset.tabular_file), str(Path(tmp_file.name).expanduser().resolve()))

        # check error is triggered if incorrect type is passed
        with self.assertRaises(FedbiomedDatasetError):
            dataset.tabular_file = 1233

        with self.assertRaises(FedbiomedDatasetError):
            dataset.tabular_file = []

        with self.assertRaises(FedbiomedDatasetError):
            dataset.tabular_file = True

        # check error is triggered if path is not existing
        with self.assertRaises(FedbiomedDatasetError):
            dataset.tabular_file = '/a/non/existing/file'

        # check error is triggered if a folder is passed instead of a file
        temp_dir = tempfile.mkdtemp()
        with self.assertRaises(FedbiomedDatasetError):
            dataset.tabular_file = temp_dir

    def test_medical_folder_dataset_06_index_col_setter(self):
        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,)

        # test with a index col string
        index_col_str = '1234'  # def _check_modality_exists(self, modality: List[str]) -> bool:

        dataset.index_col = index_col_str

        self.assertEqual(dataset.index_col, index_col_str)

        # test with a index col integer
        index_col_int = 1234
        dataset.index_col = index_col_int

        self.assertEqual(dataset.index_col, dataset.index_col)

        # check error is triggered if incorrect type is passed
        with self.assertRaises(FedbiomedDatasetError):
            dataset.index_col = 2.

        with self.assertRaises(FedbiomedDatasetError):
            dataset.index_col = [1, 2]

        with self.assertRaises(FedbiomedDatasetError):
            dataset.index_col = {}

    def test_medical_folder_dataset_07_instantiation_with_demographics(self):
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col,
                                       demographics_transform=lambda x: torch.as_tensor(x['AGE']))
        self._assert_batch_types_and_sizes(dataset)

    def test_medical_folder_dataset_08_demographics_getter(self):
        # test getter
        # # test case where tabular file is None
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file,)

        df = dataset.demographics
        self.assertIsNone(df)

        # # test case where index_col is None
        dataset = MedicalFolderDataset(self.root, index_col=self.index_col)
        df = dataset.demographics
        self.assertIsNone(df)

        # # test normal case scenario: loading demographics file
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col)
        self.assertIsInstance(dataset.demographics, pd.DataFrame)

        # # create dataset with duplicated patients, and check if duplicated values are removed
        values = {"A": [1, 2, 3, 4],
                  "B": ['patient_1', 'patient_2', 'patient_1', 'patient_3'],
                  "C": ['a', 'b', 'c', 'd']}
        df = pd.DataFrame(values)
        tmp_file = tempfile.mkdtemp()
        csv_name = os.path.join(tmp_file, 'test_csv.csv')
        df.to_csv(csv_name)

        dataset = MedicalFolderDataset(self.root, tabular_file=csv_name, index_col=2)

        values = {"A": [1, 2, 4],
                  "B": ['patient_1', 'patient_2', 'patient_3'],
                  "C": ['a', 'b', 'd']}
        demographics_without_index = dataset.demographics[["A", "C"]].reset_index(drop=True)
        # compare demograhics without index first
        self.assertTrue(demographics_without_index.equals(pd.DataFrame(values)[["A", "C"]]))
        # compare index of dataframe

        self.assertListEqual(dataset.demographics.index.tolist(), values["B"])

        # # test if error is raised when unable to load demographic file
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col)

        patcher = patch('fedbiomed.common.data._medical_datasets.MedicalFolderDataset.read_demographics',
                        side_effect=OSError)
        patcher.start()
        try:
            with self.assertRaises(FedbiomedDatasetError):
                df = dataset.demographics
        finally:
            patcher.stop()

    def test_medical_folder_dataset_09_demographics_setter(self):
        # check that it is not possible to set demographic attribute
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file,)
        with self.assertRaises(AttributeError):
            dataset.demographics = pd.DataFrame({"A": [1, 2, 3, 4], "B": ['a', 'b', 'c', 'c']})

    def test_medical_folder_dataset_10_shape(self):
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col)
        shape = dataset.shape()

        self.assertEqual(shape, {'T1': [10, 10, 10],
                                 'label': [10, 10, 10],
                                 'demographics': (10, 2),
                                 'num_modalities': 2})

        # check shape with 2 modalities + labels
        dataset = MedicalFolderDataset(self.root,
                                       tabular_file=self.tabular_file,
                                       index_col=self.index_col,
                                       data_modalities=['T1', 'T2'])

        shape = dataset.shape()
        self.assertEqual(shape, {'T1': [10, 10, 10],
                                 'T2': [10, 10, 10],
                                 'label': [10, 10, 10],
                                 'demographics': (10, 2),
                                 'num_modalities': 3})

    def test_medical_folder_dataset_11_data_transforms(self):
        dataset = MedicalFolderDataset(self.root, transform=self.transform)

        for i, ((images, demographics), targets) in enumerate(dataset):
            # test indexation
            self.assertTrue(images['T1'].dim() == 1)
            # test iteration
            (images_indxed, _), _ = dataset[i]
            self.assertTrue(images_indxed['T1'].dim() == 1)

    def test_medical_folder_dataset_12_target_transform(self):
        dataset = MedicalFolderDataset(self.root, target_transform=self.target_transform)
        (images, demographics), targets = dataset[0]
        self.assertEqual(images['T1'].shape, targets['label'].shape)

    def test_medical_folder_dataset_13_set_dataset_parameters(self):
        dataset = MedicalFolderDataset(self.root)

        with self.assertRaises(FedbiomedDatasetError):
            dataset.set_dataset_parameters("NONEDICTPARAMS")

        for params in {'bad_key': 1}, \
                      {"tabular_file": self.tabular_file, 'bad_key': 1}, \
                      {'bad_key': 1, "tabular_file": self.tabular_file, "index_col": self.index_col}:
            with self.assertRaises(FedbiomedDatasetError):
                dataset.set_dataset_parameters(params)


        dataset.set_dataset_parameters({"tabular_file": self.tabular_file, "index_col": self.index_col})
        self.assertEqual(str(dataset.tabular_file), str(Path(self.tabular_file).expanduser().resolve()))
        self.assertEqual(dataset.index_col, self.index_col)

    def test_medical_folder_dataset_14_demographics_transform(self):
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col,
                                       demographics_transform=lambda x: torch.as_tensor(x['AGE']))

        # check indexation

        csv_data = pd.read_csv(self.tabular_file)

        # check over a loop
        for i, ((images, demographics), targets) in enumerate(dataset):
            self.assertTrue(demographics.numpy() in csv_data.AGE.values)
            (images, demographics_indxed), targets = dataset[i]
            self.assertTrue(demographics_indxed.numpy() in csv_data.AGE.values)

        dataset = MedicalFolderDataset(self.root, demographics_transform=lambda x: torch.as_tensor(x['AGE']))
        with self.assertRaises(FedbiomedDatasetError):
            (images, demographics), targets = dataset[0]

        def robust_transform(demographics):
            if isinstance(demographics, dict) and len(demographics) == 0:
                return demographics
            else:
                return demographics['AGE']
        dataset = MedicalFolderDataset(self.root, tabular_file=self.tabular_file, index_col=self.index_col,
                                       demographics_transform=robust_transform)
        (images, demographics), targets = dataset[0]
        csv_data = pd.read_csv(self.tabular_file)
        self.assertTrue(demographics.numpy() in csv_data.AGE.values)

        dataset = MedicalFolderDataset(self.root, demographics_transform=robust_transform)
        (images, demographics), targets = dataset[0]
        self.assertIsInstance(demographics, torch.Tensor)
        self.assertTrue(demographics.numel() == 0)

    def _assert_batch_types_and_sizes(self, dataset: Dataset):
        """Asserts first batches correct types and lengths

        Args:
            dataset (Dataset): a Dataset object that should have
                correct types (dict, dict, torch.Tensor) and correct batch size
        Raises:
            AssertionError if test fails
        """
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        (images, demographics), targets = next(iter(data_loader))  # get the first iteration of dataloader

        self.assertIsInstance(images, dict)
        self.assertIsInstance(targets, dict)
        self.assertIsInstance(demographics, torch.Tensor)

        lengths = [len(b) for b in images.values()]
        lengths += [len(b) for b in targets.values()]
        lengths += [demographics.shape[0]]

        # Assert for batch size on modalities and demographics
        self.assertTrue(len(set(lengths)) == 1)

    @patch('pathlib.Path.iterdir', new=patch_modality_iterdir)
    @patch('pathlib.Path.is_dir', new=patch_is_modality_dir)
    @patch('pathlib.Path.glob', new=patch_modality_glob)
    def test_medical_folder_dataset_15_data_loading_plan(self):
        medical_folder_controller = MedicalFolderController(root=self.root)
        dataset = medical_folder_controller.load_MedicalFolder()
        self.assertFalse(dataset.subject_folders())

        # with DataLoadingPlan
        medical_folder_controller = MedicalFolderController(root=self.root)
        dlb = MapperBlock()
        dlb.map = modalities_to_folders
        medical_folder_controller.set_dlp(DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb}))
        dataset = medical_folder_controller.load_MedicalFolder()
        expected_subject_folders = [Path(self.root).joinpath('subj1'),
                                    Path(self.root).joinpath('subj2')]
        for p1, p2 in zip(dataset.subject_folders(), expected_subject_folders):
            self.assertEqual(os.path.realpath(p1), os.path.realpath(p2))

        dataset._reader = MagicMock()
        dataset._reader.side_effect = lambda x: x
        images_dict = dataset.load_images(Path(self.root).joinpath('subj1'), ['T1', 'T2'])
        expected_images_dict = {
            'T1': Path('T1philips_test.nii').resolve(),
            'T2': Path('T2_test.nii').resolve()
        }
        self.assertEqual(images_dict, expected_images_dict)
        images_dict = dataset.load_images(Path(self.root).joinpath('subj2'), ['T1', 'T2'])
        expected_images_dict = {
            'T1': Path('T1siemens_test.nii').resolve(),
            'T2': Path('T2_test.nii').resolve()
        }
        self.assertEqual(images_dict, expected_images_dict)
        with self.assertRaises(FedbiomedLoadingBlockError):
            _ = dataset.load_images(Path(self.root).joinpath('subj1'), ['non-existing-modality'])

        dataset = MedicalFolderDataset(root=self.root,
                                       tabular_file=None,
                                       index_col=None,
                                       data_modalities=['T1', 'T2'],
                                       target_modalities=['label'])
        dataset._reader = MagicMock()
        dataset._reader.side_effect = lambda x: x
        with self.assertRaises(FedbiomedDatasetError):
            _ = dataset[0]

        dataset.set_dlp(DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb}))
        (data, demographics), label = dataset[0]
        self.assertEqual(data, {'T1': Path('T1philips_test.nii').resolve(), 'T2': Path('T2_test.nii').resolve()})
        self.assertEqual(demographics.numel(), 0)
        self.assertEqual(label, {'label': Path('label_test.nii').resolve()})

    def test_medical_folder_dataset_16_load_MedicalFolder(self):

        # correct calls to load_MedicalFolder
        medical_folder_controller = MedicalFolderController(root=self.root)

        dataset = medical_folder_controller.load_MedicalFolder()
        self.assertTrue(isinstance(dataset, MedicalFolderDataset))
        self.assertEqual(os.path.realpath(dataset.root),
                         os.path.realpath(self.root))

        # bad call, no root defined
        medical_folder_controller = MedicalFolderController()
        with self.assertRaises(FedbiomedDatasetError):
            medical_folder_controller.load_MedicalFolder()

        # bad call, MedicalFolderDataset creation fails
        medical_folder_controller = MedicalFolderController(root=self.root)

        mfd_patcher = patch('fedbiomed.common.data.MedicalFolderDataset.__init__', side_effect=FedbiomedDatasetError)
        mfd_patcher.start()
        with self.assertRaises(FedbiomedDatasetError):
            medical_folder_controller.load_MedicalFolder()
        mfd_patcher.stop()


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

        # alternate root
        self.root2 = tempfile.mkdtemp()
        self.tabular_file2 = os.path.join(self.root2, 'participants.csv')
        _create_synthetic_dataset(self.root2, self.n_samples, self.tabular_file2, self.index_col)

    def tearDown(self) -> None:

        if 'IXI' not in self.root:
            shutil.rmtree(self.root)  # is that useful since temporary folder will be deleted
        pass

    def test_medical_folder_base_01_init(self):
        self.medical_folder_base = MedicalFolderBase()
        self.assertIsNone(self.medical_folder_base.root, "MedicalFolderBase root should not in empty initialization")

        self.medical_folder_base = MedicalFolderBase(root=self.root2)
        self.assertIsInstance(self.medical_folder_base.root, PosixPath)
        self.assertEqual(os.path.realpath(self.medical_folder_base.root),
                         os.path.realpath(self.root2),
                         "MedicalFolderBase root should not in empty initialization")

        # Setting root to a valid value with setter
        self.medical_folder_base.root = self.root

        self.assertIsInstance(self.medical_folder_base.root, PosixPath)
        self.assertEqual(os.path.realpath(self.medical_folder_base.root),
                         os.path.realpath(self.root),
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

        # create temporary file and check if MedicalFolderBase triggers error if file is passed instead a folder
        Path(os.path.join(dummy_root_2, 'test_file')).touch()
        with self.assertRaises(FedbiomedDatasetError):
            MedicalFolderBase(root=os.path.join(dummy_root_2, 'test_file'))
        # Remove tmp folder
        shutil.rmtree(dummy_root_2)

    def test_medical_folder_base_02_modalities(self):
        """Testing the method gets modalities from subject folder"""

        self.medical_folder_base = MedicalFolderBase(root=self.root)
        unique_modalities, all_modalities = self.medical_folder_base.modalities()

        self.assertIsInstance(all_modalities, list, "All modalities are not as expected")
        unique_modalities.sort()
        self.assertListEqual(unique_modalities, ["T1", "T2", "label"])

    def test_medical_folder_base_03_modalities_existing(self):
        self.medical_folder_base = MedicalFolderBase(root=self.root)
        demographics = pd.read_csv(self.tabular_file)
        for subject in demographics[self.index_col]:

            logical = all(self.medical_folder_base.is_modalities_existing(subject,
                                                                          ['T1', 'T2', 'label']))
            self.assertTrue(logical)

        # remove one modality to each subject
        for subject in demographics[self.index_col]:
            modalities_folders_path = os.listdir(os.path.join(self.root, subject))
            modality_to_remove = choice(modalities_folders_path)
            shutil.rmtree(os.path.join(self.root, subject, modality_to_remove))

            # action
            logical = self.medical_folder_base.is_modalities_existing(subject,
                                                                      ['T1', 'T2', 'label'])
            # checks
            self.assertFalse(all(logical))
            self.assertTrue(any(logical))

        # incorrect types for subject (expecting string)
        for subject in 3, {}, True, ("my subject"), { "my subject": 1}, ["my subject"]:
            with self.assertRaises(FedbiomedDatasetError):
                self.medical_folder_base.is_modalities_existing(subject, ['T1', 'T2', 'label'])

        # incorrect types for modalities (expecting list of strings)
        for modalities in "this is not a list", 3, {}, True, ("un"), { "un": 1}, [1], ['T1', 4], ['T1', 'T2', 'label', []]:
            with self.assertRaises(FedbiomedDatasetError):
                self.medical_folder_base.is_modalities_existing("any subject", modalities)

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

    def test_medical_folder_base_05_deographic_column_names(self):
        self.medical_folder_base = MedicalFolderBase(root=self.root)

        variable_names = ['var_1', 'var_2', 'var_3']

        test_csv = pd.DataFrame({v: np.random.randn(10) for v in variable_names})
        test_csv.to_csv(os.path.join(self.root, 'toto.csv'), index=False)

        # action
        col = self.medical_folder_base.demographics_column_names(os.path.join(self.root, 'toto.csv'))

        # check
        self.assertListEqual(col.tolist(), variable_names)

    @patch('pathlib.Path.iterdir', new=patch_modality_iterdir)
    @patch('pathlib.Path.is_dir', new=patch_is_modality_dir)
    @patch('pathlib.Path.glob', new=patch_modality_glob)
    def test_medical_folder_base_06_modalities_existing_multiple_names(self):
        medical_folder_base = MedicalFolderBase(root=self.root)

        self.assertEqual(
            medical_folder_base._subject_modality_folder('subj1', 'T1philips'),
            Path('T1philips'))
        self.assertIsNone(medical_folder_base._subject_modality_folder('subj1', 'T1siemens'))

        all_modalities = ['T1philips', 'T1siemens', 'T2', 'label', 'non-existing-modality']
        is_modalities_existing = medical_folder_base.is_modalities_existing('subj1', all_modalities)
        self.assertEqual(is_modalities_existing, [True, False, True, True, False])
        is_modalities_existing = medical_folder_base.is_modalities_existing('subj2', all_modalities)
        self.assertEqual(is_modalities_existing, [False, True, True, True, False])
        is_modalities_existing = medical_folder_base.is_modalities_existing('subj3', all_modalities)
        self.assertEqual(is_modalities_existing, [False, False, False, False, True])

        complete_subjects = medical_folder_base.complete_subjects(['subj1', 'subj2', 'subj3'], all_modalities)
        self.assertFalse(complete_subjects)

        # with DataLoadingPlan
        dlb = MapperBlock()
        dlb.map = modalities_to_folders
        medical_folder_base.set_dlp(DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb}))

        self.assertEqual(
            medical_folder_base._subject_modality_folder('subj1', 'T1'),
            Path('T1philips'))
        self.assertIsNone(medical_folder_base._subject_modality_folder('subj3', 'T1'))

        is_modalities_existing = medical_folder_base.is_modalities_existing('subj1', ['T1', 'T2', 'label'])
        self.assertEqual(is_modalities_existing, [True, True, True])
        is_modalities_existing = medical_folder_base.is_modalities_existing('subj2', ['T1', 'T2', 'label'])
        self.assertEqual(is_modalities_existing, [True, True, True])
        is_modalities_existing = medical_folder_base.is_modalities_existing('subj3', ['T1', 'T2', 'label'])
        self.assertEqual(is_modalities_existing, [False, False, False])

        complete_subjects = medical_folder_base.complete_subjects(['subj1', 'subj2', 'subj3'],
                                                                  ['T1', 'T2', 'label'])
        self.assertEqual(complete_subjects, ['subj1', 'subj2'])

    def test_medical_folder_base_07_subject_modality_folder(self):
        medical_folder_base = MedicalFolderBase(root=self.root)

        # calling with bad arguments
        for subject in 3, {}, True, ("my subject"), { "my subject": 1}, ["my subject"]:
            with self.assertRaises(FedbiomedDatasetError):
                medical_folder_base._subject_modality_folder(subject, "my modality")

        for modality in 3, {}, True, ("my modality"), { "my modality": 1}, ["my modality"]:
            with self.assertRaises(FedbiomedDatasetError):
                medical_folder_base._subject_modality_folder("my subject", modality)


class TestMedicalFolderController(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.tabular_file = os.path.join(self.root, 'participants.csv')
        self.index_col = 'FOLDER_NAME'
        self.n_samples = 20

        _create_synthetic_dataset(self.root, self.n_samples, self.tabular_file, self.index_col)

    def tearDown(self) -> None:
        pass

    def test_medical_folder_controller_01_subject_modality_status(self):
        medical_folder_controller = MedicalFolderController(self.root)

        # check method when index set to None
        res = medical_folder_controller.subject_modality_status()

        self.assertListEqual(sorted(res['columns']), sorted(['T1', 'T2', 'label']))
        self.assertTrue(any(res['data']))

        csv_data = pd.read_csv(self.tabular_file)
        self.assertListEqual(sorted(csv_data[self.index_col].tolist()),
                             sorted(res['index']))

        # in the folowing, we will run 5 tests with different size of patient_id
        patient_ids = csv_data[self.index_col].tolist()
        for _ in range(5):
            random.shuffle(patient_ids)
            len_sample = random.randint(1, self.n_samples)
            patient_selected = patient_ids[0: len_sample]
            res = medical_folder_controller.subject_modality_status(patient_selected)
            self.assertListEqual(sorted(patient_ids),
                                 sorted(res['index']))

    @patch('pathlib.Path.iterdir', new=patch_modality_iterdir)
    @patch('pathlib.Path.is_dir', new=patch_is_modality_dir)
    @patch('pathlib.Path.glob', new=patch_modality_glob)
    def test_medical_folder_controller_02_modalities(self):
        medical_folder_controller = MedicalFolderController(self.root)
        unique_modalities, modalities = medical_folder_controller.modalities()
        expected_unique_modalities = {'T1philips', 'label', 'T2', 'non-existing-modality', 'T1siemens'}
        expected_modalities = ['T1philips', 'T2', 'label', 'T1siemens', 'T2', 'label', 'non-existing-modality']
        self.assertEqual(set(unique_modalities), expected_unique_modalities)
        self.assertEqual(modalities, expected_modalities)

        dlb = MapperBlock()
        dlb.map = modalities_to_folders
        medical_folder_controller.set_dlp(DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb}))
        unique_modalities, modalities = medical_folder_controller.modalities()
        expected_unique_modalities = {'T1', 'T2', 'label'}
        self.assertEqual(set(unique_modalities), expected_unique_modalities)
        self.assertEqual(modalities, expected_modalities)


if __name__ == '__main__':
    unittest.main()
