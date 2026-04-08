# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import List, Optional, Union

from fedbiomed.common.exceptions import (
    FedbiomedDatasetError,
    FedbiomedDatasetManagerError,
)
from fedbiomed.common.logger import logger
from fedbiomed.node.cli_utils._io import (
    _prompt_path_cli,
    validated_data_type_input,
    validated_path_input,
)
from fedbiomed.node.cli_utils._medical_folder_dataset import (
    add_medical_folder_dataset_from_cli,
)
from fedbiomed.node.dataset_manager import DatasetManager

from ._tkinter_utils import messagebox


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


def add_database(
    dataset_manager: DatasetManager,
    interactive: bool = True,
    path: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    description: Optional[str] = None,
    data_type: Optional[str] = None,
    dataset_parameters: Optional[dict] = None,
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

    need_interactive_input = (
        interactive
        or path is None
        or name is None
        or tags is None
        or description is None
        or data_type is None
    )

    if need_interactive_input:
        print("Welcome to the Fed-BioMed CLI data manager")

        data_type = validated_data_type_input() if interactive else "default"

        _predefined = {
            "default": ("MNIST", "MNIST database", ["#MNIST", "#dataset"]),
            "mednist": ("MEDNIST", "MEDNIST dataset", ["#MEDNIST", "#dataset"]),
        }

        if data_type in _predefined:
            name, description, tags = _predefined[data_type]
            if interactive:
                tags = _confirm_predefined_dataset_tags(name, tags)
                path = validated_path_input(data_type)

        else:
            name = input("Name of the database: ")
            tags = (
                input("Tags (separate them by comma and no spaces): ")
                .replace(" ", "")
                .split(",")
            )
            description = input("Description: ")

            if data_type == "medical-folder":
                path, dataset_parameters, data_loading_plan = (
                    add_medical_folder_dataset_from_cli(
                        dataset_parameters, data_loading_plan
                    )
                )
            elif data_type == "custom":
                while True:
                    abs_path = Path(input("Path to the dataset: ")).resolve()
                    if abs_path.exists():
                        break
                    print(f"Path not found: {abs_path}. Please try again.")
                path = str(abs_path)
            else:
                path = validated_path_input(data_type)

        if interactive and data_loading_plan is not None:
            while True:
                desc = input(
                    "Please input a short name/description for your data loading plan:"
                )
                if len(desc) >= 4:
                    break
                print("Description must be at least 4 characters long.")
            data_loading_plan.desc = desc

    else:
        tags = str(tags).split(",")
        name = str(name)
        description = str(description)

        data_type = str(data_type).lower()
        if data_type not in (
            "csv",
            "default",
            "mednist",
            "images",
            "medical-folder",
            "custom",
        ):
            data_type = "default"

        if path is not None and not os.path.exists(path):
            logger.warning("provided path does not exist: " + path)
            path = _prompt_path_cli(data_type)

    if path is None:
        raise FedbiomedDatasetManagerError("Dataset path is not set")
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
        if interactive and messagebox is not None:
            messagebox.showwarning(title="Warning", message=str(e))
        else:
            logger.error(str(e))
        exit(1)
    except FedbiomedDatasetError as err:
        logger.error(
            f"{err} ... Aborting"
            "\nHint: are you sure you have selected the correct index in Demographic file?"
        )
        exit(1)

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
    d_id: Optional[str] = None

    if interactive:
        options = [d["name"] for d in my_data]
        msg = "Select the dataset to delete:\n"
        msg += "\n".join([f"{i}) {d}" for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

    while True:
        try:
            if interactive:
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
        except (ValueError, AssertionError):
            print("Invalid option. Please try again.")


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
