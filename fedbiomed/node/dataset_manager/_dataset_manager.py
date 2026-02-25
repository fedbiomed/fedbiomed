# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with the node component database.
"""

import csv
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from tabulate import tabulate  # only used for printing

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import get_controller
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._db_tables import (
    DatasetTable,
    DlbTable,
    DlpTable,
    DynamicDatasetTable,
)


class DatasetManager:
    """Interfaces with the node component database.

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """

    def __init__(self, path: str):
        """Initialize with database path."""
        self._dataset_table = DatasetTable(path)
        self._dynamic_dataset_table = DynamicDatasetTable(path)
        self._dlp_table = DlpTable(path)
        self._dlb_table = DlbTable(path)

    @property
    def dataset_table(self) -> DatasetTable:
        return self._dataset_table

    @property
    def dynamic_dataset_table(self) -> DynamicDatasetTable:
        return self._dynamic_dataset_table

    @property
    def dlp_table(self) -> DlpTable:
        return self._dlp_table

    @property
    def dlb_table(self) -> DlbTable:
        return self._dlb_table

    def get_dlp_by_id(self, dlp_id: str) -> Tuple[Optional[dict], List[dict]]:
        """Get data loading plan by ID and its associated data loading blocks."""
        dlp_metadata = self.dlp_table.get_by_id(dlp_id)

        if dlp_metadata is None:
            return None, []

        dlb_ids = list(dlp_metadata["loading_blocks"].values())
        return dlp_metadata, self.dlb_table.get_all_by_value("dlb_id", dlb_ids)

    def read_csv(
        self, csv_file: str, index_col: Union[int, None] = None
    ) -> pd.DataFrame:
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
        with open(csv_file, "r") as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            file.seek(0)
            header = 0 if sniffer.has_header(file.read()) else None

        return pd.read_csv(csv_file, index_col=index_col, sep=delimiter, header=header)

    def add_database(
        self,
        name: str,
        data_type: str,
        tags: Union[tuple, list],
        description: str,
        path: str,
        dataset_id: Optional[str] = None,
        dataset_parameters: Optional[dict] = None,
        data_loading_plan: Optional[DataLoadingPlan] = None,
        save_dlp: bool = True,
    ):
        """Register a dataset in the database.

        Args:
            name: Name of the dataset
            data_type: Type of the dataset (e.g. 'tabular', 'image-folder', 'medical-folder')
            tags: Tags associated with the dataset
            description: Description of the dataset
            path: Path to the dataset
            dataset_id: Optional ID for the dataset. If None, a new ID will be generated.
            dataset_parameters: Optional parameters for the dataset controller
            data_loading_plan: Optional DataLoadingPlan associated with the dataset
            save_dlp: Whether to save the DataLoadingPlan to the database if provided

        Returns:
            The dataset_id of the registered dataset

        Raises:
            FedbiomedError:
            - If there are conflicting tags with existing datasets
            - If the data loading plan name is invalid or not unique
            - If the data_type is not supported
        """
        controller = get_controller(
            data_type,
            controller_parameters={
                "root": path,
                "dlp": data_loading_plan,
                **(dataset_parameters if dataset_parameters is not None else {}),
            },
        )

        dataset_entry = self.dataset_table.insert(
            entry=dict(
                dataset_id=dataset_id,
                name=name,
                data_type=data_type,
                tags=tags,
                description=description,
                path=controller._controller_kwargs["root"],
                shape=controller.shape(),
                dtypes=controller.get_types(),
                dataset_parameters={
                    _k: _v
                    for _k, _v in controller._controller_kwargs.items()
                    if _k not in ["root", "dlp"]
                },
                dlp_id=None if data_loading_plan is None else data_loading_plan.dlp_id,
            )
        )

        if save_dlp:
            self.save_data_loading_plan(data_loading_plan)

        return dataset_entry

    def add_dynamic_dataset(
        self,
        path: str,
        researcher_id: str,
        experiment_id: str,
        processing_id: str,
        parent_dataset_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_parameters: Optional[dict] = None,
    ):
        """Adds a dynamic dataset to the database.

        Args:
            path: the path where the dynamic dataset files are stored.
            researcher_id: the id of the researcher who created the dynamic dataset
            experiment_id: the id of the experiment for which the dynamic dataset was created
            processing_id: the id of the processing that generated the dynamic dataset
            parent_dataset_id: the id of the parent dataset from which the dynamic dataset was derived
            name: optional name for the dynamic dataset
            tags: optional list of tags for the dynamic dataset
            description: optional description for the dynamic dataset
            dataset_id: optional id for the dynamic dataset. If None, a new id will be generated.
            dataset_parameters: optional parameters for the dataset controller, such as e.g. data type specific parameters (e.g. for medical-folder, the tabular file name within the folder)

        Returns:
            The dataset_id of the registered dynamic dataset

        Raises:
            FedbiomedError:
            - If the parent dataset is not found in the database
        """
        # Validate parent
        parent_entry = self.get_dataset_entry_by_id(parent_dataset_id)

        # Insert partial DB entry
        entry = {
            "path": path,
            "researcher_id": researcher_id,
            "experiment_id": experiment_id,
            "processing_id": processing_id,
            "parent_dataset_id": parent_dataset_id,
            "name": name,
            "tags": tags,
            "description": description,
            "dataset_id": dataset_id,
            "dataset_parameters": dataset_parameters,
        }
        dataset_id = self.dynamic_dataset_table.insert(entry)

        # Get controller
        data_type = parent_entry["data_type"]
        controller = get_controller(
            data_type,
            controller_parameters={
                "root": path,
                **(dataset_parameters if dataset_parameters is not None else {}),
            },
        )

        # Update DB entry with path, shape and dtypes
        self.dynamic_dataset_table.update_by_id(
            dataset_id,
            {
                "shape": controller.shape(),
                "dtypes": controller.get_types(),
                "data_type": data_type,
            },
        )

        return dataset_id

    def get_dataset_entry_by_id(self, dataset_id: str) -> dict:
        """
        Validates that the dataset exists and returns its DB entry.

        The dataset can be either:
            - a dataset (DatasetTable)
            - a dynamic dataset (DynamicDatasetTable)

        Args:
            dataset_id: the id of the dataset to validate

        Raises:
            FedbiomedError:
            - If dataset or dynamic dataset does not exist.
        """

        dataset_entry = self.dataset_table.get_by_id(dataset_id)
        if dataset_entry is not None:
            return dataset_entry

        dynamic_dataset_entry = self.dynamic_dataset_table.get_by_id(dataset_id)
        if dynamic_dataset_entry is not None:
            return dynamic_dataset_entry

        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Dataset with id {dataset_id} not found."
        )

    def delete_dataset_by_id(
        self,
        dataset_id: str,
        recursive: bool = False,
        force: bool = False,
    ) -> None:
        """Deletes a dataset from the database by its ID.

        Args:
            dataset_id: the ID of the dataset to delete
            recursive: whether to recursively delete all descendant dynamic datasets (if any). Defaults to False.
            force: whether to force delete the dataset if it has children, by reassigning its children to its parent dataset. Defaults to False.

        Raises:
            FedbiomedError:
            - If the dataset has children and neither recursive nor force is True
            - If no dataset is found with the given ID
        """
        dataset_entry = self.get_dataset_entry_by_id(dataset_id)
        is_dynamic = "parent_dataset_id" in dataset_entry

        children = self.dynamic_dataset_table.get_all_by_value(
            "parent_dataset_id", dataset_id
        )

        # Case 1: dataset
        if not is_dynamic:
            if children:
                if force:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Cannot force delete a dataset with children. Use recursive=True to delete subtree."
                    )

                if not recursive:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Dataset has derived dynamic datasets. Use recursive=True to delete subtree."
                    )

                self._delete_dynamic_subtree(dataset_id, is_dynamic)

            # Delete raw dataset entry (no filesystem removal)
            self.dataset_table.delete_by_id(dataset_id)
            return

        # Case 2: dynamic dataset
        if children and not recursive and not force:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset has children. Use recursive=True to delete subtree "
                "or force=True to reassign children."
            )

        if recursive:
            self._delete_dynamic_subtree(dataset_id, is_dynamic)
            return

        if force:
            parent_id = dataset_entry.get("parent_dataset_id")

            for child in children:
                self.dynamic_dataset_table.update_by_id(
                    child["dataset_id"],
                    {"parent_dataset_id": parent_id},
                )

        self.dynamic_dataset_table.delete_by_id(dataset_id)

    def _delete_dynamic_subtree(self, dataset_id: str, is_dynamic: bool) -> None:
        subtree = self.dynamic_dataset_table.collect_subtree(dataset_id)
        if not is_dynamic:
            # Remove the parent dataset
            subtree = subtree[1:]
        # Delete children first
        for dyn_id in reversed(subtree):
            self.dynamic_dataset_table.delete_by_id(dyn_id)

    def remove_dlp_by_id(self, dlp_id: str):
        """Removes a data loading plan (DLP) from the database,
        along with its associated data loading blocks (DLBs).

        Args:
            dlp_id: the DataLoadingPlan id
        """
        dlp, dlbs = self.get_dlp_by_id(dlp_id)

        if dlp is not None:
            self._dlp_table.delete_by_id(dlp_id)
            for dlb in dlbs:
                self._dlb_table.delete_by_id(dlb["dlb_id"])

    def list_my_datasets(self, verbose: bool = True) -> List[dict]:
        """Lists all datasets on the node.

        Args:
            verbose: Give verbose output. Defaults to True.

        Returns:
            All datasets in the node's database.
        """
        my_data = self.dataset_table.all()

        # Do not display dtypes
        for doc in my_data:
            doc.pop("dtypes", None)

        if verbose:
            print(tabulate(my_data, headers="keys"))

        return my_data

    def save_data_loading_plan(
        self, data_loading_plan: Optional[DataLoadingPlan]
    ) -> Union[str, None]:
        """Save a DataLoadingPlan to the database.

        Args:
            data_loading_plan: the DataLoadingPlan to be saved, or None.

        Returns:
            The `dlp_id` if a DLP was saved, otherwise None

        """
        if data_loading_plan is None:
            return None

        dlp_metadata, dlbs_metadata = data_loading_plan.serialize()
        _ = self.dlp_table.insert(dlp_metadata)
        for dlb_metadata in dlbs_metadata:
            _ = self.dlb_table.insert(dlb_metadata)
        return data_loading_plan.dlp_id

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
                    f"{ErrorNumbers.FB632.value}: Object of type {type(d)} does not "
                    "support pop or getitem method in obfuscate_private_information."
                ) from e
        return database_metadata
