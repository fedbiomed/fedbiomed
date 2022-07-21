'''
Interfaces with the node component database.
'''


import csv
import os.path
from typing import Union, List, Any, Optional
import uuid

from urllib.request import urlretrieve
from urllib.error import ContentTooShortError, HTTPError, URLError
import tarfile

from tinydb import TinyDB, Query
import pandas as pd
from tabulate import tabulate  # only used for printing

import torch
from torchvision import datasets
from torchvision import transforms

from fedbiomed.node.environ import environ

from fedbiomed.common.exceptions import FedbiomedError, FedbiomedDatasetManagerError
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.data import MedicalFolderController
from fedbiomed.common.data.data_loading_plan import DataLoadingPlan

from fedbiomed.common.logger import logger


class DatasetManager:
    """Interfaces with the node component database.

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """
    def __init__(self):
        """Constructor of the class.
        """
        self.db = TinyDB(environ['DB_PATH'])
        self.database = Query()

    def get_by_id(self, dataset_id: str) -> List[dict]:
        """Searches for data with given dataset_id.

        Args:
            dataset_id:  A dataset id

        Returns:
            A list of dict of matching datasets, each dict
                containing all the fields describing the matching datasets
                stored in Tiny database.
        """
        self.db.clear_cache()
        result = self.db.get(self.database.dataset_id == dataset_id)

        return result

    def get_dlp_by_id(self, dlp_id: str) -> dict:
        self.db.clear_cache()
        table = self.db.table('Data_Loading_Plans')
        result = table.get(self.database.dlp_id == dlp_id)
        return result

    def search_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for data with given tags.

        Args:
            tags:  List of tags

        Returns:
            The list of matching datasets
        """
        self.db.clear_cache()
        return self.db.search(self.database.tags.all(tags))

    def read_csv(self, csv_file: str, index_col: Union[int, None] = None) -> pd.DataFrame:
        """Gets content of a CSV file.

        Reads a *.csv file and outputs its data into a pandas DataFrame.
        Finds automatically the CSV delimiter by parsing the first line.

        Args:
            csv_file: File name / path
            index_col: Column that contains CSV file index.
                Defaults to None.

        Returns:
            Pandas DataFrame with data contained in CSV file.
        """

        # Automatically identify separator and header
        sniffer = csv.Sniffer()
        with open(csv_file, 'r') as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            file.seek(0)
            header = 0 if sniffer.has_header(file.read()) else None

        return pd.read_csv(csv_file, index_col=index_col, sep=delimiter, header=header)

    def get_torch_dataset_shape(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Gets info about dataset shape.

        Args:
            dataset: A Pytorch dataset

        Returns:
            A list of int containing
                [<nb_of_data>, <dimension_of_first_input_data>].
                Example for MNIST: [60000, 1, 28, 28], where <nb_of_data>=60000
                and <dimension_of_first_input_data>=1, 28, 28
        """
        return [len(dataset)] + list(dataset[0][0].shape)

    def get_csv_data_types(self, dataset: pd.DataFrame) -> List[str]:
        """Gets data types of each variable in dataset.

        Args:
            dataset: A Pandas dataset.

        Returns:
            A list of strings containing data types.
        """
        types = [str(t) for t in dataset.dtypes]

        return types

    def load_default_database(self,
                              name: str,
                              path: str,
                              as_dataset: bool = False) -> Union[List[int],
                                                                 torch.utils.data.Dataset]:
        """Loads a default dataset.

        Currently, only MNIST dataset is used as the default dataset.

        Args:
            name: Name of the default dataset. Currently,
                only MNIST is accepted.
            path: Pathfile to MNIST dataset.
            as_dataset: Whether to return
                the complete dataset (True) or dataset dimensions (False).
                Defaults to False.

        Raises:
            NotImplementedError: Name is not matching with
                the name of a default dataset.

        Returns:
            Depends on the value of the parameter `as_dataset`: If
            set to True,  returns dataset (type: torch.utils.data.Dataset).
            If set to False, returns the size of the dataset stored inside
            a list (type: List[int]).
        """
        kwargs = dict(root=path, download=True, transform=transforms.ToTensor())

        if 'mnist' in name.lower():
            dataset = datasets.MNIST(**kwargs)
        else:
            raise NotImplementedError(f'Default dataset `{name}` has'
                                      'not been implemented.')
        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)

    def load_mednist_database(self,
                              path: str,
                              as_dataset: bool = False) -> Union[List[int],
                                                                 torch.utils.data.Dataset]:
        """Loads the MedNist dataset.

        Args:
            path: Pathfile to save a local copy of the MedNist dataset.
            as_dataset: Whether to return
                the complete dataset (True) or dataset dimensions (False).
                Defaults to False.

        Raises:
            FedbiomedDatasetManagerError: One of the following cases:

                - tarfile cannot be downloaded
                - downloaded tarfile cannot
                    be extracted
                - MedNIST path is empty
                - one of the classes path is empty

        Returns:
            Depends on the value of the parameter `as_dataset`: If
            set to True,  returns dataset (type: torch.utils.data.Dataset).
            If set to False, returns the size of the dataset stored inside
            a list (type: List[int])
        """
        download_path = os.path.join(path, 'MedNIST')
        if not os.path.isdir(download_path):
            url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
            filepath = os.path.join(path, 'MedNIST.tar.gz')
            try:
                logger.info("Now downloading MEDNIST...")
                urlretrieve(url, filepath)
                with tarfile.open(filepath) as tar_file:
                    logger.info("Now extracting MEDNIST...")
                    tar_file.extractall(path)
                os.remove(filepath)

            except (URLError, HTTPError, ContentTooShortError, OSError, tarfile.TarError,
                    MemoryError) as e:
                _msg = ErrorNumbers.FB315.value + "\nThe following error was raised while downloading MedNIST dataset"\
                    + "from the MONAI repo:  " + str(e)
                logger.error(_msg)
                raise FedbiomedDatasetManagerError(_msg)

        try:
            dataset = datasets.ImageFolder(download_path,
                                           transform=transforms.ToTensor())

        except (FileNotFoundError, RuntimeError) as e:
            _msg = ErrorNumbers.FB315.value + "\nThe following error was raised while loading MedNIST dataset from"\
                "the selected path:  " + str(e) + "\nPlease make sure that the selected MedNIST folder is not empty \
                   or choose another path."
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        except Exception as e:
            _msg = ErrorNumbers.FB315.value + "\nThe following error was raised while loading MedNIST dataset" + str(e)
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)

    def load_images_dataset(self,
                            folder_path: str,
                            as_dataset: bool = False) -> Union[List[int],
                                                               torch.utils.data.Dataset]:
        """Loads an image dataset.

        Args:
            folder_path: Path to the directory containing the images.
            as_dataset: Whether to return
                the complete dataset (True) or dataset dimensions (False).
                Defaults to False.

        Returns:
            Depends on the value of the parameter `as_dataset`: If
            set to True,  returns dataset (type: torch.utils.data.Dataset).
            If set to False, returns the size of the dataset stored inside
            a list (type: List[int])
        """
        try:
            dataset = datasets.ImageFolder(folder_path,
                                           transform=transforms.ToTensor())
        except Exception as e:
            _msg = ErrorNumbers.FB315.value +\
                "\nThe following error was raised while loading dataset from the selected" \
                " path:  " + str(e) + "\nPlease make sure that the selected folder is not empty \
                and doesn't have any empty class folder"
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)

    def load_csv_dataset(self, path: str) -> pd.DataFrame:
        """Loads a CSV dataset.

        Args:
            path: Path to the CSV file.

        Returns:
            Pandas DataFrame with the content of the file.
        """
        return self.read_csv(path)

    def add_database(self,
                     name: str,
                     data_type: str,
                     tags: Union[tuple, list],
                     description: str,
                     path: str,
                     dataset_id: str = None,
                     dataset_parameters : Optional[dict] = None,
                     data_loading_plan: Optional[dict] = None):
        """Adds a new dataset contained in a file to node's database.

        Args:
            name: Name of the dataset
            data_type: File extension/format of the
                dataset (*.csv, images, ...)
            tags: Tags of the dataset.
            description: Human readable description of the dataset.
            path: Path to the dataset.
            dataset_id: Id of the dataset. Defaults to None.

        Raises:
            NotImplementedError: `data_type` is not supported.
            FedbiomedDatasetManagerError: path does not exist or dataset was not saved properly.
        """
        # Accept tilde as home folder
        path = os.path.expanduser(path)

        # Check that there are not existing databases with the same name
        assert len(self.search_by_tags(tags)) == 0, 'Data tags must be unique'

        dtypes = []  # empty list for Image datasets
        data_types = ['csv', 'default', 'mednist', 'images', 'medical-folder']
        if data_type not in data_types:
            raise NotImplementedError(f'Data type {data_type} is not'
                                      ' a compatible data type. '
                                      f'Compatible data types are: {data_types}')


        if data_type == 'default':
            assert os.path.isdir(path), f'Folder {path} for Default Dataset does not exist.'
            shape = self.load_default_database(name, path)

        elif data_type == 'mednist':
            assert os.path.isdir(path), f'Folder {path} for MedNIST Dataset does not exist.'
            shape = self.load_mednist_database(path)
            path = os.path.join(path, 'MedNIST')

        elif data_type == 'csv':
            assert os.path.isfile(path), f'Path provided ({path}) does not correspond to a CSV file.'
            dataset = self.load_csv_dataset(path)
            shape = dataset.shape
            dtypes = self.get_csv_data_types(dataset)

        elif data_type == 'images':
            assert os.path.isdir(path), f'Folder {path} for Images Dataset does not exist.'
            shape = self.load_images_dataset(path)

        elif data_type == 'medical-folder':
            if not os.path.isdir(path):
                raise FedbiomedDatasetManagerError(f'Folder {path} for Medical Folder Dataset does not exist.')

            if "tabular_file" not in dataset_parameters:
                logger.info("Medical Folder Dataset will be loaded without reference/demographics data.")
            else:
                if not os.path.isfile(dataset_parameters['tabular_file']):
                    raise FedbiomedDatasetManagerError(f'Path {dataset_parameters["tabular_file"]} does not '
                                                       f'correspond a file.')
                if "index_col" not in dataset_parameters:
                    raise FedbiomedDatasetManagerError('Index column is not provided')

            try:
                # load using the MedicalFolderController to ensure all available modalities are inspected
                controller = MedicalFolderController(root=path)
                dataset = controller.load_MedicalFolder(tabular_file=dataset_parameters.get('tabular_file', None),
                                                        index_col=dataset_parameters.get('index_col', None))
            except FedbiomedError as e:
                raise FedbiomedDatasetManagerError(f"Can not create Medical Folder dataset. {e}")
            else:
                shape = dataset.shape()

            # try to read one sample and raise if it doesn't work
            try:
                _ = dataset.get_nontransformed_item(0)
            except Exception as e:
                raise FedbiomedDatasetManagerError(f'Medical Folder Dataset was not saved properly and '
                                                   f'cannot be read. {e}')

        if not dataset_id:
            dataset_id = 'dataset_' + str(uuid.uuid4())

        new_database = dict(name=name, data_type=data_type, tags=tags,
                            description=description, shape=shape,
                            path=path, dataset_id=dataset_id, dtypes=dtypes,
                            dataset_parameters=dataset_parameters)
        new_database = self._handle_save_data_loading_plan(new_database, data_loading_plan)
        self.db.insert(new_database)

        return dataset_id

    def remove_database(self, tags: Union[tuple, list]):
        """Removes datasets from database.

        Only datasets matching the `tags` should be removed.

        Args:
            tags: Dataset description tags.
        """
        doc_ids = [doc.doc_id for doc in self.search_by_tags(tags)]
        self.db.remove(doc_ids=doc_ids)

    def modify_database_info(self,
                             tags: Union[tuple, list],
                             modified_dataset: dict):
        """Modifies a dataset in the database.

        Args:
            tags: Tags describing the dataset to modify.
            modified_dataset: New dataset description to replace the existing one.
        """
        self.db.update(modified_dataset, self.database.tags.all(tags))

    def list_my_data(self, verbose: bool = True) -> List[dict]:
        """Lists all datasets on the node.

        Args:
            verbose: Give verbose output. Defaults to True.

        Returns:
            All datasets in the node's database.
        """
        self.db.clear_cache()
        my_data = self.db.all()

        # Do not display dtypes
        for doc in my_data:
            doc.pop('dtypes')

        if verbose:
            print(tabulate(my_data, headers='keys'))

        return my_data

    def load_as_dataloader(self, dataset: dict) -> torch.utils.data.Dataset:
        """Loads content of an image dataset.

        Args:
            dataset: Description of the dataset.

        Returns:
            Content of the dataset.
        """
        name = dataset['data_type']
        if name == 'default':
            return self.load_default_database(name=dataset['name'],
                                              path=dataset['path'],
                                              as_dataset=True)
        elif name == 'images':
            return self.load_images_dataset(folder_path=dataset['path'],
                                            as_dataset=True)

    # TODO: `load_data` seems unused, prune in next refactor ?
    def load_data(self, tags: Union[tuple, list], mode: str) -> Any:
        """Loads content of a dataset.

        Args:
            tags: Tags describing the dataset to load.
            mode: Return format for the dataset content.

        Raises:
            NotImplementedError: `mode` is not implemented yet.

        Returns:
            Content of the dataset. Its type depends on the `mode` and dataset.
        """

        # Verify is mode is available
        mode = mode.lower()
        modes = ['pandas', 'torch_dataset', 'torch_tensor', 'numpy']
        if mode not in modes:
            raise NotImplementedError(f'Data mode `{mode}` was not found.'
                                      f' Data modes available: {modes}')

        # Look for dataset in database
        dataset = self.search_by_tags(tags)[0]
        print(dataset)
        assert len(dataset) > 0, f'Dataset with tags {tags} was not found.'

        dataset_path = dataset['path']
        # If path is a file, you will aim to read it with
        if os.path.isfile(dataset_path):
            df = self.read_csv(dataset_path, index_col=0)

            # Load data as requested
            if mode == 'pandas':
                return df
            elif mode == 'numpy':
                return df._get_numeric_data().values
            elif mode == 'torch_tensor':
                return torch.from_numpy(df._get_numeric_data().values)

        elif os.path.isdir(dataset_path):
            if mode == 'torch_dataset':
                return self.load_as_dataloader(dataset)
            elif mode == 'torch_tensor':
                raise NotImplementedError('We are working on this'
                                          ' implementation!')
            elif mode == 'numpy':
                raise NotImplementedError('We are working on this'
                                          'implementation!')
            else:
                raise NotImplementedError(f'Mode `{mode}` has not been'
                                          ' implemented on this version.')

    def _handle_save_data_loading_plan(self,
                                       current_dataset_metadata: dict,
                                       data_loading_plan: DataLoadingPlan
                                       ):
        if data_loading_plan is None:
            return current_dataset_metadata

        table = self.db.table('Data_Loading_Plans')
        table.insert(data_loading_plan.serialize())
        current_dataset_metadata['dlp_id'] = data_loading_plan.dlp_id
        return current_dataset_metadata
