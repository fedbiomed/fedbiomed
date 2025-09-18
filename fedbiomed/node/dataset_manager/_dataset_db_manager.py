# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Manage the node's database table for handling datasets
"""

from typing import Iterable, Optional, Union

from tabulate import tabulate

from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.exceptions import FedbiomedError

from ._db import DatasetDB
from ._dlp_db_manager import DlpDatabaseManager


class DatasetDatabaseManager:
    """Manage the node's database table for handling datasets

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """

    _dataset_table: DatasetDB
    _dlp_manager: DlpDatabaseManager

    def __init__(self, path: str):
        self._dataset_table = DatasetDB(path)
        self._dlp_manager = DlpDatabaseManager(path)

    def get_dataset_by_id(self, dataset_id: str) -> Optional[dict]:
        """Get dataset information by its ID"""
        return self._dataset_table.get_by_id(dataset_id)

    def list_dlp_by_target_dataset_type(
        self, target_dataset_type: str = None
    ) -> list[dict]:
        """Return all dlps matching the requested target type"""
        return self._dlp_manager.list_dlp_by_target_dataset_type(target_dataset_type)

    def get_dlp_by_id(self, dlp_id: str) -> tuple[dict, list[dict]]:
        """Get DLP information by its ID, along with associated DLBs"""
        return self._dlp_manager.get_dlp_by_id(dlp_id)

    def search_datasets_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for data with given tags."""
        return self._dataset_table.search_by_tags(tags)

    def search_conflicting_datasets_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for registered data that have conflicting tags with the given tags"""
        return self._dataset_table.search_conflicting_tags(tags)

    # TODO: add dataset

    def remove_dlp_by_id(self, dlp_id: str):
        """Removes DLP from the database, along with its associated DLBs."""
        self._dlp_manager.remove_dlp_by_id(dlp_id)

    def remove_dataset_by_id(self, dataset_id: str):
        """Removes a dataset from database."""
        self._dataset_table.remove_database(dataset_id)

    def update_dataset_by_id(self, dataset_id: str, modified_dataset: dict):
        """Modifies a dataset in the database."""
        self._dataset_table.modify_database_info(dataset_id, modified_dataset)

    def list_my_data(self, verbose: bool = True) -> list[dict]:
        """Lists all datasets on the node.

        Args:
            verbose: Give verbose output. Defaults to True.

        Returns:
            All datasets in the node's database.
        """
        my_data = self._dataset_table.all()

        # Do not display dtypes
        for doc in my_data:
            doc.pop("dtypes")

        if verbose:
            print(tabulate(my_data, headers="keys"))

        return my_data

    def save_data_loading_plan(
        self, data_loading_plan: Optional[DataLoadingPlan]
    ) -> Union[str, None]:
        """Save a DataLoadingPlan to the database."""
        return self._dlp_manager.save_data_loading_plan(data_loading_plan)

    @staticmethod
    def obfuscate_private_information(
        database_metadata: Iterable[dict],
    ) -> Iterable[dict]:
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
                d.pop("path", None)
                # obfuscations specific for each data type
                if "data_type" in d:
                    if d["data_type"] == "medical-folder":
                        if "dataset_parameters" in d:
                            d["dataset_parameters"].pop("tabular_file", None)
            except AttributeError as e:
                raise FedbiomedError(
                    f"Object of type {type(d)} does not support pop or getitem method "
                    f"in obfuscate_private_information."
                ) from e
        return database_metadata
