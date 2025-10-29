# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import tkinter.messagebox
import warnings
from importlib import import_module
from typing import Union

from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.exceptions import (
    FedbiomedDatasetError,
    FedbiomedDatasetManagerError,
)
from fedbiomed.common.logger import logger
from fedbiomed.node.cli_utils._io import validated_data_type_input, validated_path_input
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
)
from fedbiomed.node.dataset_manager import DatasetManager


def _confirm_predefined_dataset_tags(dataset_name: str, default_tags: list) -> list:
    """Interactively confirm or customize tags for predefined datasets.

    Args:
        dataset_name: Name of the predefined dataset
        default_tags: Default tags for the dataset

    Returns:
        list: Final tags for the dataset (either default or user-provided)
    """
    while True:
        response = input(f"{dataset_name} will be added with tags {default_tags} [Y/n]")
        response = response.lower().strip()
        # default to yes on empty input
        if response in ["y", "yes", ""]:
            return default_tags
        elif response in ["n", "no"]:
            tags = input("Tags (separate them by comma and no spaces): ")
            return tags.replace(" ", "").split(",")
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def _handle_flamby_dataset_input():
    """Helper function to handle FLamby dataset input collection.

    Returns:
        tuple: (path, data_loading_plan) for the FLamby dataset
    """
    from fedbiomed.common.dataset.flamby_dataset import (
        FlambyDatasetMetadataBlock,
        FlambyLoadingBlockTypes,
        discover_flamby_datasets,
    )

    # Select the type of dataset (fed_ixi, fed_heart, etc...)
    available_flamby_datasets = discover_flamby_datasets()
    msg = "Please select the FLamby dataset that you're configuring:\n"
    msg += "\n".join([f"\t{i}) {val}" for i, val in available_flamby_datasets.items()])
    msg += "\nselect: "
    keep_asking_for_input = True
    while keep_asking_for_input:
        try:
            flamby_dataset_index = input(msg)
            flamby_dataset_index = int(flamby_dataset_index)
            # check that the user inserted a number within the valid range
            if flamby_dataset_index in available_flamby_datasets.keys():
                keep_asking_for_input = False
            else:
                warnings.warn(
                    f"Please pick a number in the range {list(available_flamby_datasets.keys())}",
                    stacklevel=1,
                )
        except ValueError:
            warnings.warn(
                "Please input a numeric value (integer)",
                stacklevel=1,
            )

    path = available_flamby_datasets[
        flamby_dataset_index
    ]  # flamby datasets not identified by their path

    # Select the center id
    module = import_module(
        f".{available_flamby_datasets[flamby_dataset_index]}",
        package="flamby.datasets",
    )
    n_centers = module.NUM_CLIENTS
    keep_asking_for_input = True
    while keep_asking_for_input:
        try:
            center_id = int(
                input(f"Give a center id between 0 and {str(n_centers - 1)}: ")
            )
            if 0 <= center_id < n_centers:
                keep_asking_for_input = False
        except ValueError:
            warnings.warn(
                f"Please input a numeric value (integer) between 0 and {str(n_centers - 1)}",
                stacklevel=1,
            )

    # Build the DataLoadingPlan with the selected dataset type and center id
    data_loading_plan = DataLoadingPlan()
    metadata_dlb = FlambyDatasetMetadataBlock()
    metadata_dlb.metadata = {
        "flamby_dataset_name": available_flamby_datasets[flamby_dataset_index],
        "flamby_center_id": center_id,
    }
    data_loading_plan[FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA] = metadata_dlb

    return path, data_loading_plan


def add_database(
    dataset_manager: DatasetManager,
    interactive: bool = True,
    path: str = None,
    name: str = None,
    tags: str = None,
    description: str = None,
    data_type: str = None,
    dataset_parameters: dict = None,
) -> None:
    """Adds a dataset to the node database.

    Also queries interactively the user on the command line (and file browser)
    for dataset parameters if needed.

    Args:
        dataset_manager: Object for managing the dataset
        interactive: Whether to query interactively for dataset parameters
            even if they are all passed as arguments. Defaults to `True`.
        path: Path to the dataset.
        name: Keyword for the dataset.
        tags: Comma separated list of tags for the dataset.
        description: Human readable description of the dataset.
        data_type: Keyword for the data type of the dataset.
        dataset_parameters: Parameters for the dataset manager
    """

    data_loading_plan = None

    # if all args are provided, just try to load the data
    # if not, ask the user more information
    need_interactive_input = (
        interactive
        or path is None
        or name is None
        or tags is None
        or description is None
        or data_type is None
    )

    if need_interactive_input:
        # Interactive mode: collect dataset parameters from user
        print("Welcome to the Fed-BioMed CLI data manager")

        # Determine data type
        if interactive:
            data_type = validated_data_type_input()
        else:
            data_type = "default"

        if data_type == "default":
            name = "MNIST"
            description = "MNIST database"
            tags = ["#MNIST", "#dataset"]
            if interactive is True:
                tags = _confirm_predefined_dataset_tags(name, tags)
                path = validated_path_input(data_type)

        elif data_type == "mednist":
            name = "MEDNIST"
            description = "MEDNIST dataset"
            tags = ["#MEDNIST", "#dataset"]
            if interactive is True:
                tags = _confirm_predefined_dataset_tags(name, tags)
                path = validated_path_input(data_type)

        # Handle custom datasets
        else:
            # Collect dataset metadata
            name = input("Name of the database: ")
            tags = input("Tags (separate them by comma and no spaces): ")
            tags = tags.replace(" ", "").split(",")
            description = input("Description: ")

            if data_type == "medical-folder":
                path, dataset_parameters, data_loading_plan = (
                    add_medical_folder_dataset_from_cli(
                        dataset_parameters, data_loading_plan
                    )
                )
            elif data_type == "flamby":
                path, data_loading_plan = _handle_flamby_dataset_input()
            else:
                path = validated_path_input(data_type)

        # if a data loading plan was specified, we now ask for the description
        if interactive and data_loading_plan is not None:
            keep_asking_for_input = True
            while keep_asking_for_input:
                desc = input(
                    "Please input a short name/description for your data loading plan:"
                )
                if len(desc) < 4:
                    print("Description must be at least 4 characters long.")
                else:
                    keep_asking_for_input = False
            data_loading_plan.desc = desc

    else:
        # Non-interactive mode:
        # all data have been provided at call
        # check few things

        # transform a string with comma(s) as a string list
        tags = str(tags).split(",")
        name = str(name)
        description = str(description)

        # Validate data type
        data_type = str(data_type).lower()
        if data_type not in ["csv", "default", "mednist", "images", "medical-folder"]:
            data_type = "default"

        # Validate path
        if not os.path.exists(path):
            logger.critical("provided path does not exists: " + path)

    # Ensure path is absolute
    path = os.path.abspath(path)
    logger.info(f"Dataset absolute path: {path}")

    try:
        dataset_manager.add_database(
            name=name,
            tags=tags,
            data_type=data_type,
            description=description,
            path=path,
            dataset_parameters=dataset_parameters,
            data_loading_plan=data_loading_plan,
        )
    except (AssertionError, FedbiomedDatasetManagerError) as e:
        if interactive is True:
            try:
                tkinter.messagebox.showwarning(title="Warning", message=str(e))
            except ModuleNotFoundError:
                warnings.warn(f"[ERROR]: {e}", stacklevel=1)
        else:
            warnings.warn(f"[ERROR]: {e}", stacklevel=1)
        exit(1)
    except FedbiomedDatasetError as err:
        warnings.warn(
            f"[ERROR]: {err} ... Aborting"
            "\nHint: are you sure you have selected the correct index in Demographic file?",
            stacklevel=1,
        )

    # Display success message
    print("\nGreat! Take a look at your data:")
    dataset_manager.list_my_datasets(verbose=True)


def delete_database(dataset_manager: DatasetManager, interactive: bool = True) -> None:
    """Removes one or more dataset from the node's database.

    Does not modify the dataset's files.

    Args:
        interactive:

            - if `True` interactively queries (repeatedly) from the command line
                for a dataset to delete
            - if `False` delete MNIST dataset if it exists in the database
    """
    my_data = dataset_manager.list_my_datasets(verbose=False)
    if not my_data:
        logger.warning("No dataset to delete")
        return

    msg: str = ""
    d_id: Union[str, None] = None

    if interactive is True:
        options = [d["name"] for d in my_data]
        msg = "Select the dataset to delete:\n"
        msg += "\n".join([f"{i}) {d}" for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

    while True:
        try:
            if interactive is True:
                opt_idx = int(input(msg)) - 1
                assert opt_idx in range(len(my_data))

                d_id = my_data[opt_idx]["dataset_id"]
            else:
                for ds in my_data:
                    if ds["name"] == "MNIST":
                        d_id = ds["dataset_id"]
                        break

            if not d_id:
                logger.warning("No matching dataset to delete")
                return
            dataset_manager.dataset_table.delete_by_id(d_id)
            logger.info("Dataset removed. Here your available datasets")
            dataset_manager.list_my_datasets()
            return
        except (ValueError, IndexError, AssertionError):
            logger.error("Invalid option. Please, try again.")


def delete_all_database(dataset_manager: DatasetManager) -> None:
    """Deletes all datasets from the node's database.

    Does not modify the dataset's files.

    Args:
        dataset_manager: Object for managing the dataset
    """
    my_data = dataset_manager.list_my_datasets(verbose=False)

    if not my_data:
        logger.warning("No dataset to delete")
        return

    for ds in my_data:
        d_id = ds["dataset_id"]
        dataset_manager.dataset_table.delete_by_id(d_id)
        logger.info("Dataset removed for dataset_id:" + str(d_id))
