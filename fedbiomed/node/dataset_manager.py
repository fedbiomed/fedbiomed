# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
Interfaces with the node component database.
'''


import csv
import os.path
from typing import Iterable, Union, List, Optional, Tuple
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

from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedDatasetManagerError
from fedbiomed.common.constants import ErrorNumbers, DatasetTypes
from fedbiomed.common.data import MedicalFolderController, DataLoadingPlan, DataLoadingBlock
from fedbiomed.common.logger import logger


class DatasetManager:
    """Interfaces with the node component database.

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """
    def __init__(self, db: str):
        """Constructor of the class.

        Args:
            db: Path to the database file
        """
        self._db = TinyDB(db)
        self._database = Query()

        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._dataset_table = DBTable(self._db.storage, name='Datasets', cache_size=0)
        self._dlp_table = DBTable(self._db.storage, name='Data_Loading_Plans', cache_size=0)

    def get_by_id(self, dataset_id: str) -> Union[dict, None]:
        """Searches for a dataset with given dataset_id.

        Args:
            dataset_id:  A dataset id

        Returns:
            A `dict` containing the dataset's description if a dataset with this `dataset_id`
            exists in the database. `None` if no such dataset exists in the database.
        """
        return self._dataset_table.get(self._database.dataset_id == dataset_id)


    def list_dlp(self, target_dataset_type: Optional[str] = None) -> List[dict]:
        """Return all existing DataLoadingPlans.

        Args:
            target_dataset_type: (str or None) if specified, return only dlps matching the requested target type.

        Returns:
            An array of dict, each dict is a DataLoadingPlan
        """
        if target_dataset_type is not None:
            if not isinstance(target_dataset_type, str):
                raise FedbiomedDatasetManagerError(f"Wrong input type for target_dataset_type. "
                                                   f"Expected str, got {type(target_dataset_type)} instead.")
            if target_dataset_type not in [t.value for t in DatasetTypes]:
                raise FedbiomedDatasetManagerError("target_dataset_type should be of the values defined in "
                                                   "fedbiomed.common.constants.DatasetTypes")

            return self._dlp_table.search(
                (self._database.dlp_id.exists()) &
                (self._database.dlp_name.exists()) &
                (self._database.target_dataset_type == target_dataset_type))
        else:
            return self._dlp_table.search(
                (self._database.dlp_id.exists()) & (self._database.dlp_name.exists()))

    def get_dlp_by_id(self, dlp_id: str) -> Tuple[dict, List[dict]]:
        """Search for a DataLoadingPlan with a given id.

        Note that in case of conflicting ids (which should not happen), this function will silently return a random
        one with the sought id.

        DataLoadingPlan IDs always start with 'dlp_' and should be unique in the database.

        Args:
            dlp_id: (str) the DataLoadingPlan id

        Returns:
            A Tuple containing a dictionary with the DataLoadingPlan metadata corresponding to the given id.
        """
        dlp_metadata = self._dlp_table.get(self._database.dlp_id == dlp_id)

        # TODO: This exception should be removed once non-existing DLP situation is
        # handled by higher layers in Round or Node classes
        if dlp_metadata is None:
            raise FedbiomedDatasetManagerError(
                f"{ErrorNumbers.FB315.value}: Non-existing DLP for the dataset."
            )

        return dlp_metadata, self._dlp_table.search(
            self._database.dlb_id.one_of(dlp_metadata['loading_blocks'].values()))

    def get_data_loading_blocks_by_ids(self, dlb_ids: Union[str, List[str]]) -> List[dict]:
        """Search for a list of DataLoadingBlockTypes, each corresponding to one given id.

        Note that in case of conflicting ids (which should not happen), this function will silently return a random
        one with the sought id.

        DataLoadingBlock IDs always start with 'serialized_data_loading_block_' and should be unique in the database.

        Args:
            dlb_ids: (List[str]) a list of DataLoadingBlock IDs

        Returns:
            A list of dictionaries, each one containing the DataLoadingBlock metadata corresponding to one given id.
        """
        return self._dlp_table.search(self._database.dlb_id.one_of(dlb_ids))

    def search_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for data with given tags.

        Args:
            tags:  List of tags

        Returns:
            The list of matching datasets
        """
        return self._dataset_table.search(self._database.tags.all(tags))

    def search_conflicting_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for registered data that have conflicting tags with the given tags

        Args:
            tags:  List of tags

        Returns:
            The list of conflicting datasets
        """
        def _conflicting_tags(val):
            return all(t in val for t in tags) or all(t in tags for t in val)


        return self._dataset_table.search(self._database.tags.test(_conflicting_tags))

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
                              as_dataset: bool = False) -> Tuple[Union[List[int],
                                                            torch.utils.data.Dataset], str]:
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
            Tuple of 2 items:
            First item Depends on the value of the parameter `as_dataset`: If
            set to True,  returns dataset (type: torch.utils.data.Dataset).
            If set to False, returns the size of the dataset stored inside
            a list (type: List[int])
            Second item is the path used to download the MedNIST dataset, that needs to be saved as an
            entry in the dataset
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
            return dataset, download_path
        else:
            return self.get_torch_dataset_shape(dataset), download_path

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
                     path: Optional[str] = None,
                     dataset_id: Optional[str] = None,
                     dataset_parameters : Optional[dict] = None,
                     data_loading_plan: Optional[DataLoadingPlan] = None,
                     save_dlp: bool = True):
        """Adds a new dataset contained in a file to node's database.

        Args:
            name: Name of the dataset
            data_type: File extension/format of the
                dataset (*.csv, images, ...)
            tags: Tags of the dataset.
            description: Human readable description of the dataset.
            path: Path to the dataset. Defaults to None.
            dataset_id: Id of the dataset. Defaults to None.
            dataset_parameters: a dictionary of additional (customized) parameters, or None
            data_loading_plan: a DataLoadingPlan to be linked to this dataset, or None
            save_dlp: if True, save the `data_loading_plan`

        Returns:
            dataset_id: id of the dataset stored in database. Returns `dataset_id`
                if provided (non-None) or a new id if not.

        Raises:
            NotImplementedError: `data_type` is not supported.
            FedbiomedDatasetManagerError: path does not exist or dataset was not saved properly.
        """
        # Accept tilde as home folder
        if path is not None:
            path = os.path.expanduser(path)

        # Check that there are not existing dataset with conflicting tags
        conflicting = self.search_conflicting_tags(tags)
        if len(conflicting) > 0:
            msg = f"{ErrorNumbers.FB322.value}, one or more registered dataset has conflicting tags: " \
                f" {' '.join([ c['name'] for c in conflicting ])}"
            logger.critical(msg)
            raise FedbiomedDatasetManagerError(msg)

        dtypes = []  # empty list for Image datasets
        data_types = ['csv', 'default', 'mednist', 'images', 'medical-folder', 'flamby']

        if data_type not in data_types:
            raise NotImplementedError(f'Data type {data_type} is not'
                                      ' a compatible data type. '
                                      f'Compatible data types are: {data_types}')

        elif data_type == 'flamby':
            from fedbiomed.common.data.flamby_dataset import FlambyLoadingBlockTypes, FlambyDataset
            # check that data loading plan is present and well formed
            if data_loading_plan is None or \
                    FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA not in data_loading_plan:
                msg = f"{ErrorNumbers.FB316.value}. A DataLoadingPlan containing " \
                      f"{FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA.value} is required for adding a FLamby dataset " \
                      f"to the database."
                logger.critical(msg)
                raise FedbiomedDatasetManagerError(msg)

            # initialize a dataset and link to the flamby data. If all goes well, compute shape.
            try:
                dataset = FlambyDataset()
                dataset.set_dlp(data_loading_plan)  # initializes fed_class as a side effect
            except FedbiomedError as e:
                raise FedbiomedDatasetManagerError(f"Can not create FLamby dataset. {e}")
            else:
                shape = dataset.shape()

        if data_type == 'default':
            assert os.path.isdir(path), f'Folder {path} for Default Dataset does not exist.'
            shape = self.load_default_database(name, path)

        elif data_type == 'mednist':
            assert os.path.isdir(path), f'Folder {path} for MedNIST Dataset does not exist.'
            shape, path = self.load_mednist_database(path)

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
                if data_loading_plan is not None:
                    controller.set_dlp(data_loading_plan)
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
        if save_dlp:
            dlp_id = self.save_data_loading_plan(data_loading_plan)
        elif isinstance(data_loading_plan, DataLoadingPlan):
            dlp_id = data_loading_plan.dlp_id
        else:
            dlp_id = None
        if dlp_id is not None:
            new_database['dlp_id'] = dlp_id

        self._dataset_table.insert(new_database)

        return dataset_id

    def remove_dlp_by_id(self, dlp_id: str):
        """Removes a data loading plan (DLP) from the database.

        Only DLP with matching ID is removed from the database. There should be at most one.

        If `remove_dlbs` is True, also remove the attached DLBs. You should ensure
        they are not used by another DLP, no verification is made.

        Args:
            dlp_id: the DataLoadingPlan id
        """
        if not isinstance(dlp_id, str):
            _msg = ErrorNumbers.FB316.value + f": Bad type for dlp '{type(dlp_id)}', expecting str"
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)
        if not str:
            _msg = ErrorNumbers.FB316.value + ": Bad value for dlp, expecting non empty str"
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        _ , dlbs = self.get_dlp_by_id(dlp_id)
        try:
            self._dlp_table.remove(self._database.dlp_id == dlp_id)
            for dlb in dlbs:
                self._dlp_table.remove(self._database.dlb_id == dlb['dlb_id'])
        except Exception as e:
            _msg = ErrorNumbers.FB316.value + f": Error during remove of DLP {dlp_id}: {e}"
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

    def remove_database(self, dataset_id: str):
        """Removes a dataset from database.

        Only the dataset matching the `dataset_id` should be removed.

        Args:
            dataset_id: Dataset unique ID.
        """
        # TODO: check that there is no more than one dataset with `dataset_id` (consistency, should not happen)
        _, dataset_document = self._dataset_table.get(self._database.dataset_id == dataset_id, add_docs=True)

        if dataset_document:
            self._dataset_table.remove(doc_ids=[dataset_document.doc_id])
        else:
            _msg = ErrorNumbers.FB322.value + f": No dataset found with id {dataset_id}"
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

    def modify_database_info(self,
                             dataset_id: str,
                             modified_dataset: dict):
        """Modifies a dataset in the database.

        Args:
            dataset_id: ID of the dataset to modify.
            modified_dataset: New dataset description to replace the existing one.

        Raises:
            FedbiomedDatasetManagerError: conflicting tags with existing dataset
        """
        # Check that there are not existing dataset with conflicting tags
        if 'tags' in modified_dataset:
            conflicting = self.search_conflicting_tags(modified_dataset['tags'])

            conflicting_ids = [ c['dataset_id'] for c in conflicting ]
            # the dataset to modify is ignored (can conflict with its previous tags)
            if dataset_id in conflicting_ids:
                conflicting_ids.remove(dataset_id)

            if len(conflicting_ids) > 0:
                msg = f"{ErrorNumbers.FB322.value}, one or more registered dataset has conflicting tags: " \
                    f" {' '.join([ c['name'] for c in conflicting if c['dataset_id'] != dataset_id ])}"
                logger.critical(msg)
                raise FedbiomedDatasetManagerError(msg)

        self._dataset_table.update(modified_dataset, self._database.dataset_id == dataset_id)

    def list_my_data(self, verbose: bool = True) -> List[dict]:
        """Lists all datasets on the node.

        Args:
            verbose: Give verbose output. Defaults to True.

        Returns:
            All datasets in the node's database.
        """
        my_data = self._dataset_table.all()

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

    def save_data_loading_plan(self,
                               data_loading_plan: Optional[DataLoadingPlan]
                               ) -> Union[str, None]:
        """Save a DataLoadingPlan to the database.

        This function saves a DataLoadingPlan to the database, and returns its ID.

        Raises:
            FedbiomedDatasetManagerError: bad data loading plan name (size, not unique)

        Args:
            data_loading_plan: the DataLoadingPlan to be saved, or None.

        Returns:
            The `dlp_id` if a DLP was saved, or None
        """
        if data_loading_plan is None:
            return None

        if len(data_loading_plan.desc) < 4:
            _msg = ErrorNumbers.FB316.value + ": Cannot save data loading plan, " + \
                "DLP name needs to have at least 4 characters."
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        _dlp_same_name = self._dlp_table.search(
            (self._database.dlp_id.exists()) & (self._database.dlp_name.exists()) &
            (self._database.dlp_name == data_loading_plan.desc))
        if _dlp_same_name:
            _msg = ErrorNumbers.FB316.value + ": Cannot save data loading plan, " + \
                "DLP name needs to be unique."
            logger.error(_msg)
            raise FedbiomedDatasetManagerError(_msg)

        dlp_metadata, loading_blocks_metadata = data_loading_plan.serialize()
        self._dlp_table.insert(dlp_metadata)
        self._dlp_table.insert_multiple(loading_blocks_metadata)
        return data_loading_plan.dlp_id

    def save_data_loading_block(self, dlb: DataLoadingBlock) -> None:
        # seems unused
        self._dlp_table.insert(dlb.serialize())

    @staticmethod
    def obfuscate_private_information(database_metadata: Iterable[dict]) -> Iterable[dict]:
        """Remove privacy-sensitive information, to prepare for sharing with a researcher.

        Removes any information that could be considered privacy-sensitive by the node. The typical use-case is to
        prevent sharing this information with a researcher through a reply message.

        Args:
            database_metadata: an iterable of metadata information objects, one per dataset. Each metadata object
                should be in the format af key-value pairs, such as e.g. a dict.
        Returns:
             the updated iterable of metadata information objects without privacy-sensitive information
        """
        for d in database_metadata:
            try:
                # common obfuscations
                d.pop('path', None)
                # obfuscations specific for each data type
                if 'data_type' in d:
                    if d['data_type'] == 'medical-folder':
                        if 'dataset_parameters' in d:
                            d['dataset_parameters'].pop('tabular_file', None)
            except AttributeError:
                raise FedbiomedDatasetManagerError(f"Object of type {type(d)} does not support pop or getitem method "
                                                   f"in obfuscate_private_information.")
        return database_metadata
