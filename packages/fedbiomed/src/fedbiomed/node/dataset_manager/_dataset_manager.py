# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with the node component database.
"""

import csv
from typing import Iterable, List, Literal, Optional, Tuple, Union, get_args

import pandas as pd
from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session
from tabulate import tabulate  # only used for printing

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import get_controller
from fedbiomed.common.dataset_controller import download_mnist
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.node.db_entries import (
    DatasetEntry,
    DlbEntry,
    DlpEntry,
    DynamicDatasetEntry,
)


class DatasetManager:
    """Registers the node's datasets, their data loading plans and derived
    datasets, and gives access to them.

    Each public method runs in one transaction of the node's database.
    """

    # Datasets a lookup covers: registered, derived by processing jobs, or both
    DatasetScope = Literal["original", "dynamic", "both"]

    # Tables this manager owns; the only ones it reads or writes
    MODELS = (DatasetEntry, DynamicDatasetEntry, DlpEntry, DlbEntry)

    def __init__(self, db_path: str, min_samples: int = 0):
        """Initialize with the node's database, creating the missing `MODELS` tables.

        Args:
            db_path: Path of the node's database file, as in its config.
            min_samples: Minimum number of samples required when adding a dataset.
                Defaults to 0 (no minimum enforced).
        """
        self._database = get_database(db_path)
        self._database.create_tables(self.MODELS)
        self._min_samples = min_samples

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

    def get_dlp_by_id(self, dlp_id: str) -> Tuple[Optional[dict], List[dict]]:
        """Get a data loading plan and its associated data loading blocks by ID.

        Args:
            dlp_id: ID of the data loading plan to retrieve.

        Returns:
            A tuple of (dlp_metadata, dlbs) where dlp_metadata is the DLP dict
            (or None if not found) and dlbs is a list of associated DLB dicts.
        """
        with self._database.transaction() as session:
            dlp = session.get(DlpEntry, dlp_id)

            if dlp is None:
                logger.debug(
                    "Data loading plan with id %s not found in database.",
                    dlp_id,
                )
                return None, []

            logger.debug(
                "Data loading plan with id %s found in database. Retrieving associated data loading blocks.",
                dlp_id,
            )
            dlb_ids = list(dlp.loading_blocks.values())
            logger.debug(
                "Found %d data loading blocks associated with data loading plan id %s: dlb_ids=%s",
                len(dlb_ids),
                dlp_id,
                dlb_ids,
            )
            dlbs = session.scalars(select(DlbEntry).where(DlbEntry.dlb_id.in_(dlb_ids)))
            return dlp.to_dict(), [dlb.content for dlb in dlbs]

    def read_csv(
        self,
        csv_file: str,
        index_col: Union[int, None] = None,
        dtype: Union[type, None] = None,
    ) -> pd.DataFrame:
        """Gets content of a CSV file.

        Reads a *.csv file and outputs its data into a pandas DataFrame.
        Finds automatically the CSV delimiter by parsing the first line.

        Args:
            csv_file: File name / path
            index_col: Column that contains CSV file index.
                Defaults to None.
            dtype: Data type to force column read types. Defaults to None, letting
                pandas infer the type of each column.

        Returns:
            Pandas DataFrame with data contained in CSV file.
        """

        # Automatically identify separator and header
        sniffer = csv.Sniffer()
        with open(csv_file, "r") as file:
            delimiter = sniffer.sniff(file.readline()).delimiter
            file.seek(0)
            header = 0 if sniffer.has_header(file.read()) else None

        return pd.read_csv(
            csv_file, index_col=index_col, sep=delimiter, header=header, dtype=dtype
        )

    def add_database(
        self,
        name: str,
        data_type: str,
        tags: List[str],
        description: str,
        path: str,
        dataset_id: Optional[str] = None,
        dataset_parameters: Optional[dict] = None,
        data_loading_plan: Optional[DataLoadingPlan] = None,
    ) -> str:
        """Register a dataset in the database.

        The data loading plan, if any, must already be saved
        (`save_data_loading_plan`).

        Args:
            name: Name of the dataset
            data_type: Type of the dataset, a `DatasetTypes` value
                (e.g. 'csv', 'images', 'medical-folder')
            tags: Tags associated with the dataset
            description: Description of the dataset
            path: Path to the dataset
            dataset_id: Optional ID for the dataset. If None, a new ID will be generated.
            dataset_parameters: Optional parameters for the dataset controller
            data_loading_plan: Optional DataLoadingPlan associated with the dataset

        Returns:
            The dataset_id of the registered dataset

        Raises:
            FedbiomedError:
            - If there are conflicting tags with existing datasets
            - If the data loading plan is not saved
            - If the data_type is not supported
            - If MNIST is deployed and its files can neither be found nor fetched
        """
        # MNIST is the only dataset Fed-BioMed can fetch, and deploying it is the
        # only moment that is wanted; an explicit `download` overrides.
        if data_type == DatasetTypes.DEFAULT.value and (dataset_parameters or {}).get(
            "download", True
        ):
            download_mnist(path)

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
        dataset = DatasetEntry.from_dict(entry)

        with self._database.transaction() as session:
            conflicting = self._conflicting_datasets(session, dataset.tags)
            if conflicting:
                msg = (
                    f"{ErrorNumbers.FB322.value}: One or more registered datasets present "
                    f"tags conflicting with your entry. Conflicting dataset names: "
                    f"{', '.join([dataset.name for dataset in conflicting])}."
                )
                logger.error(msg)
                raise FedbiomedError(msg)

            session.add(dataset)
            return dataset.dataset_id

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
    ) -> str:
        """Adds a dynamic dataset to the database.

        Args:
            path: the path where the dynamic dataset files are stored.
            researcher_id: the id of the researcher who created the dynamic dataset
            experiment_id: the id of the experiment for which the dynamic dataset was created
            processing_id: the id of the processing that generated the dynamic dataset
            parent_dataset_id: the id of the dataset or dynamic dataset from which the dynamic dataset was derived
            name: optional name for the dynamic dataset
            tags: optional list of tags for the dynamic dataset
            description: optional description for the dynamic dataset
            dataset_id: optional id for the dynamic dataset. If None, a new id will be generated.
            dataset_parameters: optional parameters for the dataset controller, such as e.g. data type specific parameters (e.g. for medical-folder, the tabular file name within the folder)

        Returns:
            The dataset_id of the registered dynamic dataset

        Raises:
            FedbiomedError:
            - If the parent dataset is not found in the database, or is deleted
              while the dynamic dataset is loading
        """
        with self._database.transaction() as session:
            parent = self._get_or_raise(session, parent_dataset_id)
            data_type = parent.data_type
            parent_field = (
                "parent_dynamic_id"
                if isinstance(parent, DynamicDatasetEntry)
                else "parent_dataset_id"
            )

        # Loading the dataset is slow: done outside the transactions
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
                parent_field: parent_dataset_id,
            },
        )

        with self._database.transaction() as session:
            dataset = DynamicDatasetEntry.from_dict(entry)
            session.add(dataset)
            return dataset.dataset_id

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
            data_type: Dataset type, a `DatasetTypes` value (e.g. 'csv', 'images',
                'medical-folder').
            path: Filesystem root path of the dataset.
            dataset_id: ID to assign; if None, one is generated on insert.
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

    def get_dataset(
        self, dataset_id: str, scope: DatasetScope = "original"
    ) -> Optional[dict]:
        """Gets a dataset entry by its id.

        Args:
            dataset_id: the id of the dataset
            scope: which datasets to look in, a `DatasetScope` value; with
                `both`, registered datasets are looked up first.

        Returns:
            The dataset entry, or None if no dataset in `scope` has this id.

        Raises:
            FedbiomedError: scope is not a `DatasetScope` value.
        """
        if scope not in get_args(self.DatasetScope):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: scope should be one of "
                f"{get_args(self.DatasetScope)}"
            )

        with self._database.transaction() as session:
            dataset = self._find(session, dataset_id, scope=scope)
            return None if dataset is None else dataset.to_dict()

    def delete_dataset_by_id(self, dataset_id: str, recursive: bool = False) -> None:
        """Deletes a dataset or a dynamic dataset by its ID.

        The datasets derived from it, at any depth, are deleted with it only
        when `recursive` is set.

        Args:
            dataset_id: the ID of the dataset to delete
            recursive: whether to also delete the datasets derived from it.
                Defaults to False.

        Raises:
            FedbiomedError:
            - If no dataset is found with the given ID
            - If datasets are derived from it and recursive is not set
        """
        with self._database.transaction() as session:
            dataset = self._get_or_raise(session, dataset_id)
            derived = session.scalars(
                select(DynamicDatasetEntry)
                .where(DynamicDatasetEntry.dataset_id.in_(self._derived_ids(dataset)))
                .order_by(DynamicDatasetEntry.dataset_id)
            ).all()

            if derived and not recursive:
                msg = (
                    f"{ErrorNumbers.FB632.value}: Dataset '{dataset.name or dataset_id}' "
                    "has derived datasets: "
                    f"{', '.join(repr(d.name or d.dataset_id) for d in derived)}. "
                    "Delete them too with recursive=True."
                )
                logger.error(msg)
                raise FedbiomedError(msg)

            # One statement, so the derived datasets' links are checked once
            # they are all gone
            session.execute(
                delete(DynamicDatasetEntry).where(
                    DynamicDatasetEntry.dataset_id.in_(self._derived_ids(dataset))
                )
            )
            session.delete(dataset)

    def remove_dlp_by_id(self, dlp_id: str) -> None:
        """Remove a data loading plan (DLP) and its data loading blocks (DLBs).

        Nothing happens if the DLP does not exist.

        Args:
            dlp_id: ID of the DataLoadingPlan to remove.

        Raises:
            FedbiomedError: datasets use the DLP.
        """
        with self._database.transaction() as session:
            dlp = session.get(DlpEntry, dlp_id)
            if dlp is None:
                return

            users = session.scalars(
                select(DatasetEntry.name).where(DatasetEntry.dlp_id == dlp_id)
            ).all()
            if users:
                msg = (
                    f"{ErrorNumbers.FB316.value}: Cannot remove data loading plan "
                    f"{dlp_id}: it is used by datasets: {', '.join(users)}."
                )
                logger.error(msg)
                raise FedbiomedError(msg)

            session.execute(
                delete(DlbEntry).where(
                    DlbEntry.dlb_id.in_(list(dlp.loading_blocks.values()))
                )
            )
            session.delete(dlp)

    def list_my_datasets(self, verbose: bool = True) -> List[dict]:
        """Lists all datasets on the node.

        Args:
            verbose: Give verbose output. Defaults to True.

        Returns:
            All datasets in the node's database.
        """
        with self._database.transaction() as session:
            my_data = [d.to_dict() for d in session.scalars(select(DatasetEntry))]

        # Do not display dtypes
        for entry in my_data:
            entry.pop("dtypes", None)

        if verbose:
            print(tabulate(my_data, headers="keys"))

        return my_data

    def search_datasets(self, text: str) -> List[dict]:
        """Searches datasets whose name or description contains a text.

        The search ignores case, accented letters included.

        Args:
            text: text to look for

        Returns:
            The matching datasets, each once, as in `list_my_datasets`.
        """
        wanted = text.lower()
        return [
            d
            for d in self.list_my_datasets(verbose=False)
            if wanted in d["name"].lower() or wanted in d["description"].lower()
        ]

    def search_by_tags(self, tags: Union[tuple, list]) -> List[dict]:
        """Searches for datasets having all the given tags.

        Args:
            tags: List of tags

        Returns:
            The list of matching datasets
        """
        with self._database.transaction() as session:
            datasets = session.scalars(select(DatasetEntry))
            return [d.to_dict() for d in datasets if set(tags).issubset(d.tags)]

    def search_conflicting_tags(self, tags: Union[tuple, list]) -> List[dict]:
        """Searches for registered datasets whose tags conflict with the given tags.

        Tags conflict when one set includes the other.

        Args:
            tags: List of tags

        Returns:
            The list of conflicting datasets
        """
        with self._database.transaction() as session:
            return [d.to_dict() for d in self._conflicting_datasets(session, tags)]

    def update_dataset(self, dataset_id: str, changes: dict) -> dict:
        """Modifies a registered dataset.

        Args:
            dataset_id: ID of the dataset to modify.
            changes: fields to replace in the dataset.

        Returns:
            The updated dataset entry.

        Raises:
            FedbiomedError:
            - no dataset with this id
            - a change targets an unknown field or the dataset id
            - the new tags conflict with another dataset
        """
        with self._database.transaction() as session:
            dataset = session.get(DatasetEntry, dataset_id)
            if dataset is None:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Dataset with id {dataset_id} not found."
                )

            if changes.get("dataset_id", dataset_id) != dataset_id:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Cannot change the field 'dataset_id'"
                )

            unknown = set(changes) - set(DatasetEntry.__table__.columns.keys())
            if unknown:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Invalid data input for Dataset: "
                    f"unknown fields {sorted(unknown)}"
                )

            for field, value in changes.items():
                setattr(dataset, field, value)

            if "tags" in changes:
                # the dataset to modify can conflict with its previous tags
                conflicting = [
                    d
                    for d in self._conflicting_datasets(session, dataset.tags)
                    if d.dataset_id != dataset_id
                ]
                if conflicting:
                    msg = (
                        f"{ErrorNumbers.FB322.value}: One or more registered datasets "
                        f"are conflicting with your new tags. Conflicting dataset names: "
                        f"{', '.join([d.name for d in conflicting])}."
                    )
                    logger.error(msg)
                    raise FedbiomedError(msg)

            return dataset.to_dict()

    def list_dlps(self, target_dataset_type: str) -> List[dict]:
        """Lists the data loading plans for a type of dataset.

        Args:
            target_dataset_type: a `DatasetTypes` value

        Returns:
            The data loading plans targeting this type of dataset.

        Raises:
            FedbiomedError: target_dataset_type is not a `DatasetTypes` value.
        """
        if target_dataset_type not in DatasetTypes.list():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: target_dataset_type should be one of "
                f"{DatasetTypes.list()}"
            )

        with self._database.transaction() as session:
            dlps = session.scalars(
                select(DlpEntry).where(
                    DlpEntry.target_dataset_type == target_dataset_type
                )
            )
            return [dlp.to_dict() for dlp in dlps]

    def save_data_loading_plan(self, data_loading_plan: DataLoadingPlan) -> str:
        """Save a new DataLoadingPlan and its loading blocks to the database.

        Args:
            data_loading_plan: the DataLoadingPlan to be saved.

        Returns:
            The `dlp_id` of the plan.

        Raises:
            FedbiomedError: the plan name is invalid or used by another plan, or
                the plan or one of its blocks is already saved.
        """
        dlp_metadata, dlbs_metadata = data_loading_plan.serialize()

        with self._database.transaction() as session:
            same_name = session.scalars(
                select(DlpEntry.dlp_id).where(
                    DlpEntry.dlp_name == dlp_metadata["dlp_name"]
                )
            ).first()
            if same_name is not None:
                msg = (
                    f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                    "DLP name needs to be unique."
                )
                logger.error(msg)
                raise FedbiomedError(msg)

            session.add(DlpEntry.from_dict(dlp_metadata))
            session.add_all(
                DlbEntry(dlb_id=dlb["dlb_id"], content=dlb) for dlb in dlbs_metadata
            )
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

    @staticmethod
    def _find(
        session: Session, dataset_id: str, scope: str
    ) -> Optional[Union[DatasetEntry, DynamicDatasetEntry]]:
        """Finds a dataset in the tables of a valid `scope`, registered datasets first."""
        # Keys are the DatasetScope values
        models = {
            "original": [DatasetEntry],
            "dynamic": [DynamicDatasetEntry],
            "both": [DatasetEntry, DynamicDatasetEntry],
        }[scope]
        for model in models:
            dataset = session.get(model, dataset_id)
            if dataset is not None:
                return dataset
        return None

    def _get_or_raise(
        self, session: Session, dataset_id: str
    ) -> Union[DatasetEntry, DynamicDatasetEntry]:
        """Finds a dataset or a dynamic dataset, raising if neither exists."""
        dataset = self._find(session, dataset_id, scope="both")
        if dataset is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset with id {dataset_id} not found."
            )
        return dataset

    @staticmethod
    def _derived_ids(dataset: Union[DatasetEntry, DynamicDatasetEntry]) -> Select:
        """Query of the ids of the dynamic datasets derived from a dataset, at any depth."""
        link = (
            DynamicDatasetEntry.parent_dynamic_id
            if isinstance(dataset, DynamicDatasetEntry)
            else DynamicDatasetEntry.parent_dataset_id
        )
        tree = (
            select(DynamicDatasetEntry.dataset_id)
            .where(link == dataset.dataset_id)
            .cte(recursive=True)
        )
        tree = tree.union_all(
            select(DynamicDatasetEntry.dataset_id).join(
                tree, DynamicDatasetEntry.parent_dynamic_id == tree.c.dataset_id
            )
        )
        return select(tree.c.dataset_id)

    @staticmethod
    def _conflicting_datasets(
        session: Session, tags: Union[tuple, list]
    ) -> List[DatasetEntry]:
        """Registered datasets whose tags include the given tags, or are included in them."""
        wanted = set(tags)
        return [
            dataset
            for dataset in session.scalars(select(DatasetEntry))
            if wanted.issubset(dataset.tags) or wanted.issuperset(dataset.tags)
        ]
