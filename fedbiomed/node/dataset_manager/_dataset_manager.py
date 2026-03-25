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
from fedbiomed.common.logger import logger
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

    def __init__(self, path: str, min_samples: int = 0):
        """Initialize with database path.

        Args:
            path: Path to the database file.
            min_samples: Minimum number of samples required when adding a dataset.
                Defaults to 0 (no minimum enforced).
        """
        self._min_samples = min_samples
        self._dataset_table = DatasetTable(path)
        self._dynamic_dataset_table = DynamicDatasetTable(path)
        self._dlp_table = DlpTable(path)
        self._dlb_table = DlbTable(path)

    def validate_samples(self, n_samples: int) -> None:
        """Raise FedbiomedError if n_samples is below the configured minimum.

        Args:
            n_samples: Number of samples in the dataset to validate.

        Raises:
            FedbiomedError: If n_samples is below the configured minimum.
        """
        if self._min_samples > 0 and n_samples < self._min_samples:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset has {n_samples} samples, "
                f"which is below the node's minimum required ({self._min_samples})."
            )

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
        """Get a data loading plan and its associated data loading blocks by ID.

        Args:
            dlp_id: ID of the data loading plan to retrieve.

        Returns:
            A tuple of (dlp_metadata, dlbs) where dlp_metadata is the DLP dict
            (or None if not found) and dlbs is a list of associated DLB dicts.
        """
        dlp_metadata = self.dlp_table.get_by_id(dlp_id)

        if dlp_metadata is None:
            logger.debug(
                "Data loading plan with id %s not found in database.",
                dlp_id,
            )
            return None, []

        logger.debug(
            "Data loading plan with id %s found in database. Retrieving associated data loading blocks.",
            dlp_id,
        )
        dlb_ids = list(dlp_metadata["loading_blocks"].values())
        logger.debug(
            "Found %d data loading blocks associated with data loading plan id %s: dlb_ids=%s",
            len(dlb_ids),
            dlp_id,
            dlb_ids,
        )
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
        entry = self._build_dataset_entry(
            data_type=data_type,
            path=path,
            dataset_id=dataset_id,
            dataset_parameters=dataset_parameters,
            name=name,
            tags=tags,
            description=description,
            data_loading_plan=data_loading_plan,
        )
        dataset_entry = self.dataset_table.insert(entry)

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
        parent_entry, _ = self.get_dataset_entry_by_id(parent_dataset_id)

        # Get data type from parent dataset
        data_type = parent_entry["data_type"]

        # Build entry
        entry = self._build_dataset_entry(
            data_type=data_type,
            path=path,
            dataset_id=dataset_id,
            dataset_parameters=dataset_parameters,
            name=name,
            tags=tags,
            description=description,
            data_loading_plan=None,  # DLP is not supported for dynamic datasets for now
            extra_fields={
                "researcher_id": researcher_id,
                "experiment_id": experiment_id,
                "processing_id": processing_id,
                "parent_dataset_id": parent_dataset_id,
            },
        )
        dataset_id = self.dynamic_dataset_table.insert(entry)

        return dataset_id

    def _build_dataset_entry(
        self,
        *,
        data_type: str,
        path: str,
        dataset_id: Optional[str],
        dataset_parameters: Optional[dict],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        data_loading_plan: Optional[DataLoadingPlan] = None,
        extra_fields: Optional[dict] = None,
    ) -> dict:
        """Build the database entry dict for a dataset by instantiating its controller.

        Instantiates the appropriate dataset controller to resolve the path, shape,
        and dtypes, then assembles the fields to store in the database.

        Args:
            data_type: Dataset type string (e.g. 'tabular', 'medical-folder', 'custom').
            path: Filesystem root path of the dataset.
            dataset_id: ID to assign; if None, the table will generate one on insert.
            dataset_parameters: Extra controller kwargs (e.g. tabular_file for medical-folder).
            name: Human-readable name for the dataset.
            tags: Tags used by researchers to search for this dataset.
            description: Free-text description of the dataset.
            data_loading_plan: Optional DLP to attach to the controller and record.
            extra_fields: Additional key-value pairs to merge into the entry (e.g. for
                dynamic datasets: researcher_id, experiment_id, processing_id).

        Returns:
            Dict ready to be inserted into the dataset table.

        Raises:
            FedbiomedError: If the data_type is unsupported or sample count is below minimum.
        """
        controller_params = {
            "root": path,
            **(dataset_parameters or {}),
        }

        if data_loading_plan is not None:
            controller_params["dlp"] = data_loading_plan

        controller = get_controller(
            data_type,
            controller_parameters=controller_params,
        )

        # Custom datasets don't expose a fixed sample count, so skip validation.
        if data_type != "custom":
            self.validate_samples(len(controller))

        # Creating common entry fields
        entry = {
            "dataset_id": dataset_id,
            "data_type": data_type,
            "name": name,
            "tags": tags,
            "description": description,
            "path": controller._controller_kwargs["root"],
            "shape": controller.shape(),
            "dtypes": controller.get_types(),
            # Exclude 'root' (stored separately as 'path') and 'dlp' (stored via dlp_id).
            "dataset_parameters": {
                k: v
                for k, v in controller._controller_kwargs.items()
                if k not in ["root", "dlp"]
            },
        }

        if data_loading_plan is not None:
            entry["dlp_id"] = data_loading_plan.dlp_id
        # Adding extra fields
        if extra_fields:
            entry.update(extra_fields)

        return entry

    def get_dataset_entry_by_id(self, dataset_id: str) -> Tuple[dict, str]:
        """
        Validates that the dataset exists and returns its DB entry and dataset type.

        The dataset can be either:
            - a dataset (DatasetTable)
            - a dynamic dataset (DynamicDatasetTable)

        Args:
            dataset_id: the id of the dataset to validate

        Returns:
            Tuple of (entry dict, table name string) identifying where the record lives.

        Raises:
            FedbiomedError: If dataset or dynamic dataset does not exist.
        """

        dataset_entry = self.dataset_table.get_by_id(dataset_id)
        if dataset_entry is not None:
            return dataset_entry, self.dataset_table._table_name

        dynamic_dataset_entry = self.dynamic_dataset_table.get_by_id(dataset_id)
        if dynamic_dataset_entry is not None:
            return dynamic_dataset_entry, self.dynamic_dataset_table._table_name

        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Dataset with id {dataset_id} not found."
        )

    def delete_dataset_by_id(
        self,
        dataset_id: str,
        recursive: bool = False,
        reassign_children: bool = False,
    ) -> None:
        """Deletes a dataset from the database by its ID.

        Args:
            dataset_id: the ID of the dataset to delete
            recursive: whether to recursively delete all descendant dynamic datasets (if any). Defaults to False.
            reassign_children: whether to delete the dataset, and reassign children to the parent dataset if the dataset has children. Defaults to False.

        Raises:
            FedbiomedError:
            - If no dataset is found with the given ID
            - If the dataset is a raw dataset and has children, and reassign_children is True (to avoid reassigning children of a raw dataset)
            - If the dataset is a raw dataset and has children, and recursive is not True
            - If the dynamic dataset has children and neither recursive nor reassign_children is True
        """
        dataset_entry, dataset_type = self.get_dataset_entry_by_id(dataset_id)
        is_dynamic = dataset_type == self.dynamic_dataset_table._table_name

        children = self.dynamic_dataset_table.get_all_by_value(
            "parent_dataset_id", dataset_id
        )

        # Case 1: dataset
        if not is_dynamic:
            if children:
                if reassign_children:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Cannot reassign children of a dataset with children. Use recursive=True to delete subtree."
                    )

                if not recursive:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Dataset has derived dynamic datasets. Use recursive=True to delete subtree."
                    )

                self._delete_dynamic_subtree(dataset_id)

            # Delete raw dataset entry
            self.dataset_table.delete_by_id(dataset_id)
            return

        # Case 2: dynamic dataset
        if children and not recursive and not reassign_children:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset has children. Use recursive=True to delete subtree "
                "or reassign_children=True to reassign children."
            )

        if recursive:
            self._delete_dynamic_subtree(dataset_id)

        if reassign_children:
            parent_id = dataset_entry.get("parent_dataset_id")

            for child in children:
                self.dynamic_dataset_table.update_by_id(
                    child["dataset_id"],
                    {"parent_dataset_id": parent_id},
                )

        self.dynamic_dataset_table.delete_by_id(dataset_id)

    def _delete_dynamic_subtree(self, dataset_id: str) -> None:
        """Helper method to delete a subtree of dynamic datasets rooted at dataset_id.

        The root dataset can be either:
            - a dataset (DatasetTable)
            - a dynamic dataset (DynamicDatasetTable)

        Args:
            dataset_id: the ID of the root of the subtree to delete
        """
        subtree = self.dynamic_dataset_table.collect_subtree(dataset_id)
        # Delete leaf-first (reversed BFS/DFS order) so foreign-key constraints are respected.
        for dyn_id in reversed(subtree):
            self.dynamic_dataset_table.delete_by_id(dyn_id)

    def remove_dlp_by_id(self, dlp_id: str) -> None:
        """Remove a data loading plan (DLP) and its associated data loading blocks (DLBs).

        Args:
            dlp_id: ID of the DataLoadingPlan to remove.
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
            The `dlp_id` if a DLP was saved, otherwise None.
        """
        if data_loading_plan is None:
            return None

        dlp_metadata, dlbs_metadata = data_loading_plan.serialize()
        self.dlp_table.insert(dlp_metadata)
        for dlb_metadata in dlbs_metadata:
            self.dlb_table.insert(dlb_metadata)
        return data_loading_plan.dlp_id

    @staticmethod
    def obfuscate_private_information(
        database_metadata: Iterable[dict],
    ) -> Iterable[dict]:
        """Remove privacy-sensitive information, to prepare for sharing with a researcher.

        Removes any information that could be considered privacy-sensitive by the node. The typical use-case is to
        prevent sharing this information with a researcher through a reply message.

        Args:
            database_metadata: Iterable of metadata dicts, one per dataset.

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
