import csv
import os.path
from typing import Union
import uuid

from tinydb import TinyDB, Query
import pandas as pd
from tabulate import tabulate  # only used for printing
import torch
from torchvision import datasets
from torchvision import transforms

from fedbiomed.node.environ import DB_PATH


class Data_manager: # should this be in camelcase (smthg like DataManager)?

    def __init__(self):
        """ The constrcutor of the class
        """        
        self.db = TinyDB(DB_PATH)
        self.database = Query()

    def search_by_id(self, dataset_id: str) -> list:
        """this method searches for data with given dataset_id

        Args:
            dataset_id (str):  dataset id

        Returns:
            [list]: list of matching datasets
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

    def read_csv(self, csv_file: str, index_col:int=0) -> pd.DataFrame:
        """[summary]

        Args:
            csv_file (str): [description]
            index_col (int, optional): [description]. Defaults to 0.

        Returns:
            pd.DataFrame: [description]
        """        

        # Automatically identify separator
        first_line = open(csv_file, 'r').readline()

        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(first_line).delimiter

        return pd.read_csv(csv_file, index_col=index_col, sep=delimiter)


    def get_torch_dataset_shape(self, dataset):
        """[summary]

        Args:
            dataset ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return [len(dataset)] + list(dataset[0][0].shape)


    def load_default_database(self, name: str, path: str,
                              as_dataset: bool=False):
        """[summary]

        Args:
            name (str): [description]
            path ([type]): [description]
            as_dataset (bool, optional): [description]. Defaults to False.

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        kwargs = dict(root=path, download=True, transform=transforms.ToTensor())

        if 'mnist' in name.lower():
            dataset = datasets.MNIST(**kwargs)
        else:
            raise NotImplementedError(f'Default dataset `{name}` has not been implemented.')
        # FIXME: ` NotImplementedError` should not be raised for that (dataset not found)
        # FIXES: create a custom error for that purpose
        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)


    def load_images_dataset(self, folder_path, as_dataset=False):
        """[summary]

        Args:
            folder_path ([type]): [description]
            as_dataset (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """        

        dataset = datasets.ImageFolder(folder_path, transform=transforms.ToTensor())
        if as_dataset:
            return dataset
        else:
            return self.get_torch_dataset_shape(dataset)


    def load_csv_dataset(self, path) -> pd.DataFrame:
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self.read_csv(path).shape


    def add_database(self,
                     name: str,
                     data_type: str,
                     tags: Union[tuple, list],
                    description: str,
                    path: str,
                    dataset_id: str=None):
        # Accept tilde as home folder
        path = os.path.expanduser(path)

        # Check that there are not existing databases with the same name
        assert len(self.search_by_tags(tags)) == 0, 'Data tags must be unique'

        data_types = ['csv', 'default', 'images']
        if data_type not in data_types:
            raise NotImplementedError(f'Data type {data_type} is not a compatible data type. '
                                      f'Compatible data types are: {data_types}')

        if data_type == 'default':
            assert os.path.isdir(path), f'Folder {path} for Default Dataset does not exist.'
            shape = self.load_default_database(name, path)
        elif data_type == 'csv':
            assert os.path.isfile(path), f'Path provided ({path}) does not correspond to a CSV file.'
            shape = self.load_csv_dataset(path)
        elif data_type == 'images':
            assert os.path.isdir(path), f'Folder {path} for Images Dataset does not exist.'
            shape = self.load_images_dataset(path)

        if not dataset_id:
            dataset_id = 'dataset_' + str(uuid.uuid4())

        new_database = dict(name=name, data_type=data_type, tags=tags,
                        description=description, shape=shape, path=path, dataset_id=dataset_id)
        self.db.insert(new_database)


    def remove_database(self, tags: Union[tuple, list]):
        doc_ids = [doc.doc_id for doc in self.search_by_tags(tags)]
        self.db.remove(doc_ids=doc_ids)


    def modify_database_info(self, tags: Union[tuple, list], modified_dataset: dict):
        self.db.update(modified_dataset, self.database.tags.all(tags))


    def list_my_data(self, verbose=True):
        """[summary]

        Args:
            verbose (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """     
        self.db.clear_cache()    
        my_data = self.db.all()
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
            return self.load_default_database(name=dataset['name'], path=dataset['path'], as_dataset=True)
        elif name == 'images':
            return self.load_images_dataset(folder_path=dataset['path'], as_dataset=True)


    def load_data(self, tags: Union[tuple, list], mode: str):
        """[summary]

        Args:
            tags (Union[tuple, list]): [description]
            mode (str): [description]

        Raises:
            NotImplementedError: if mode is not in ['pandas', 'torch_dataset', 'torch_tensor', 'numpy']
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
            raise NotImplementedError(f'Data mode `{mode}` was not found. Data modes available: {modes}')

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
                raise NotImplementedError('We are working on this implementation!')
            elif mode == 'numpy':
                raise NotImplementedError('We are working on this implementation!')
            else:
                raise NotImplementedError(f'Mode `{mode}` has not been implemented on this version.')

