import csv
import os.path
from typing import Union, List
import uuid

from tinydb import TinyDB, Query
import pandas as pd
from tabulate import tabulate  # only used for printing
import torch
from torchvision import datasets
from torchvision import transforms

from fedbiomed.node.environ import environ


class DataManager:
    """Interface over TinyDB database.
    Facility fot storing, retrieving data and get data info
    on the data stored into TinyDB database.
    """
    def __init__(self):
        """ The constrcutor of the class
        """
        self.db = TinyDB(environ['DB_PATH'])
        self.database = Query()

    def search_by_id(self, dataset_id: str) -> List[dict]:
        """this method searches for data with given dataset_id

        Args:
            dataset_id (str):  dataset id

        Returns:
            [List[dict]]: list of dict of matching datasets, each dict
            containing all the fields describing the matching datasets
            stored in Tiny database.
        """
        self.db.clear_cache()
        return self.db.search(self.database.dataset_id.all(dataset_id))

    def search_by_tags(self, tags: Union[tuple, list]) -> list:
        """this method searches for data with given tags

        Args:
            tags (Union[tuple, list]):  list of tags

        Returns:
            [list]: list of matching datasets
        """
        self.db.clear_cache()
        return self.db.search(self.database.tags.all(tags))

    def read_csv(self, csv_file: str, index_col: Union[int, None] = None ) -> pd.DataFrame:
        """Reads a *.csv file and ouptuts its data into a pandas DataFrame.
        Finds automatically the csv delimiter by parsing the first line.

        Args:
            csv_file (str): file name / path
            index_col (int, optional): column that contains csv file index.
            Set to None if not present. Defaults to 0.

        Returns:
            pd.DataFrame: data contained in csv file.
        """

        # Automatically identify separator and header
        sniffer = csv.Sniffer()
        with open(csv_file, 'r') as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            file.seek(0)
            header = 0 if sniffer.has_header(file.read()) else None

        return pd.read_csv(csv_file, index_col=index_col, sep=delimiter, header=header)

    def get_torch_dataset_shape(self,
                                dataset: torch.utils.data.Dataset) -> List[int]:
        """Gets info about dataset shape.

        Args:
            dataset (torch.utils.data.Dataset): a Pytorch dataset

        Returns:
            List[int]: returns a list containing:
            [<nb_of_data>, <dimension_of_first_input_data>].
            Example for MNIST: [60000, 1, 28, 28], where <nb_of_data>=60000
            and <dimension_of_first_input_data>=1, 28, 28
        """
        return [len(dataset)] + list(dataset[0][0].shape)

    def get_csv_data_types(self, dataset: pd.DataFrame) -> List[str]:

        """Gets data types of each variable in dataset.

        Args:
            dataset (pd.DataFrame): a Pandas dataset

        Returns:
            List[int]: returns a list containing data types
        """
        types = [str(t) for t in dataset.dtypes]

        return types

    def load_default_database(self,
                              name: str,
                              path: str,
                              as_dataset: bool = False) -> Union[List[int],
                                                                 torch.utils.data.Dataset]:
        """Loads a default dataset. Currently, only MNIST dataset
        is used as the default dataset.

        Args:
            name (str): name of the default dataset. Currently,
            only MNIST is accepted.
            path (str): pathfile to MNIST dataset.
            as_dataset (bool, optional): whether to return
            the complete dataset (True) or dataset dimensions (False).
            Defaults to False.

        Raises:
            NotImplementedError: triggered if name is not matching with
            the name of a default dataset.

        Returns:
            [type]: depending on the value of the parameter `as_dataset`. If
            set to True,  returns dataset (type: torch.utils.data.Dataset),
            if set to False, returns the size of the dataset stored inside
            a list (type: List[int])
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

    def load_images_dataset(self,
                            folder_path: str,
                            as_dataset: bool = False) -> Union[List[int],
                                                               torch.utils.data.Dataset]:
        """[summary]

        Args:
            folder_path ([type]): [description]
            as_dataset (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        dataset = datasets.ImageFolder(folder_path,
                                       transform=transforms.ToTensor())
        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)

    def load_csv_dataset(self, path) -> pd.DataFrame:
        """[summary]

        Args:
            path (str): CSV path

        Returns:
            [pd.DataFrame]: Returns pandas DataFrame
        """
        return self.read_csv(path)

    def add_database(self,
                     name: str,
                     data_type: str,
                     tags: Union[tuple, list],
                     description: str,
                     path: str,
                     dataset_id: str = None):
        """
        Adds a new dataset contained in a file to node

        Args:
            name (str): [description]
            data_type (str): file extension/format of the
            dataset (*.csv, images, ...)
            tags (Union[tuple, list]): [description]
            description (str): [description]
            path (str): [description]
            dataset_id (str, optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        # Accept tilde as home folder
        path = os.path.expanduser(path)

        # Check that there are not existing databases with the same name
        assert len(self.search_by_tags(tags)) == 0, 'Data tags must be unique'

        dtypes = []  # empty list for Image datasets
        data_types = ['csv', 'default', 'images']
        if data_type not in data_types:
            raise NotImplementedError(f'Data type {data_type} is not'
                                      ' a compatible data type. '
                                      f'Compatible data types are: {data_types}')


        if data_type == 'default':
            assert os.path.isdir(path), f'Folder {path} for Default Dataset does not exist.'
            shape = self.load_default_database(name, path)
        elif data_type == 'csv':
            assert os.path.isfile(path), f'Path provided ({path}) does not correspond to a CSV file.'
            dataset = self.load_csv_dataset(path)
            shape = dataset.shape
            dtypes = self.get_csv_data_types(dataset)
        elif data_type == 'images':
            assert os.path.isdir(path), f'Folder {path} for Images Dataset does not exist.'
            shape = self.load_images_dataset(path)

        if not dataset_id:
            dataset_id = 'dataset_' + str(uuid.uuid4())

        new_database = dict(name=name, data_type=data_type, tags=tags,
                            description=description, shape=shape,
                            path=path, dataset_id=dataset_id, dtypes=dtypes)
        self.db.insert(new_database)

        return dataset_id

    def remove_database(self, tags: Union[tuple, list]):
        doc_ids = [doc.doc_id for doc in self.search_by_tags(tags)]
        self.db.remove(doc_ids=doc_ids)

    def modify_database_info(self,
                             tags: Union[tuple, list],
                             modified_dataset: dict):
        self.db.update(modified_dataset, self.database.tags.all(tags))

    def list_my_data(self, verbose: bool = True):
        """[summary]

        Args:
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        self.db.clear_cache()
        my_data = self.db.all()

        # Do not display dtypes
        for doc in my_data:
            doc.pop('dtypes')

        if verbose:
            print(tabulate(my_data, headers='keys'))

        return my_data

    def load_as_dataloader(self, dataset):
        """[summary]

        Args:
            dataset ([type]): [description]

        Returns:
            [type]: [description]
        """
        name = dataset['data_type']
        if name == 'default':
            return self.load_default_database(name=dataset['name'],
                                              path=dataset['path'],
                                              as_dataset=True)
        elif name == 'images':
            return self.load_images_dataset(folder_path=dataset['path'],
                                            as_dataset=True)

    #  `load_data` seems unused
    def load_data(self, tags: Union[tuple, list], mode: str):
        """[summary]

        Args:
            tags (Union[tuple, list]): [description]
            mode (str): [description]

        Raises:
            NotImplementedError: if mode is not in ['pandas', 'torch_dataset',
            'torch_tensor', 'numpy']
            NotImplementedError: [description]
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            [type]: [description]
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
