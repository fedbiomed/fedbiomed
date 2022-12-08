# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import tkinter.messagebox
import warnings
from importlib import import_module
from fedbiomed.common import data

from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedDatasetManagerError
from fedbiomed.common.logger import logger
from fedbiomed.common.data import MedicalFolderController, DataLoadingPlan, FlambyDatasetMetadataBlock, \
    FlambyLoadingBlockTypes
from fedbiomed.node.cli_utils._medical_folder_dataset import add_medical_folder_dataset_from_cli
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.cli_utils._io import validated_data_type_input, validated_path_input
from fedbiomed.common.data import discover_flamby_datasets


dataset_manager = DatasetManager()


def add_database(interactive: bool = True,
                 path: str = None,
                 name: str = None,
                 tags: str = None,
                 description: str = None,
                 data_type: str = None):
    """Adds a dataset to the node database.

    Also queries interactively the user on the command line (and file browser)
    for dataset parameters if needed.

    Args:
        interactive: Whether to query interactively for dataset parameters
            even if they are all passed as arguments. Defaults to `True`.
        path: Path to the dataset.
        name: Keyword for the dataset.
        tags: Comma separated list of tags for the dataset.
        description: Human readable description of the dataset.
        data_type: Keyword for the data type of the dataset.
    """

    dataset_parameters = None
    data_loading_plan = None

    # if all args are provided, just try to load the data
    # if not, ask the user more informations
    if interactive or \
            path is None or \
            name is None or \
            tags is None or \
            description is None or \
            data_type is None :


        print('Welcome to the Fed-BioMed CLI data manager')

        if interactive is True:
            data_type = validated_data_type_input()
        else:
            data_type = 'default'

        if data_type == 'default':
            tags = ['#MNIST', "#dataset"]
            if interactive is True:
                while input(f'MNIST will be added with tags {tags} [y/N]').lower() != 'y':
                    pass
                path = validated_path_input(data_type)
            name = 'MNIST'
            description = 'MNIST database'

        elif data_type == 'mednist':
            tags = ['#MEDNIST', "#dataset"]
            if interactive is True:
                while input(f'MEDNIST will be added with tags {tags} [y/N]').lower() != 'y':
                    pass
                path = validated_path_input(data_type)
            name = 'MEDNIST'
            description = 'MEDNIST dataset'
        else:

            name = input('Name of the database: ')

            tags = input('Tags (separate them by comma and no spaces): ')
            tags = tags.replace(' ', '').split(',')

            description = input('Description: ')

            if data_type == 'medical-folder':
                path, dataset_parameters, data_loading_plan = add_medical_folder_dataset_from_cli(interactive,
                                                                                                  dataset_parameters,
                                                                                                  data_loading_plan)
            elif data_type == 'flamby':
                path = None  # flamby datasets are not identified by their path

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
                            warnings.warn(f"Please pick a number in the range {list(available_flamby_datasets.keys())}")
                    except ValueError:
                        warnings.warn('Please input a numeric value (integer)')

                # Select the center id
                module = import_module(f".{available_flamby_datasets[flamby_dataset_index]}", package='flamby.datasets')
                n_centers = module.NUM_CLIENTS
                keep_asking_for_input = True
                while keep_asking_for_input:
                    try:
                        center_id = int(input(f"Give a center id between 0 and {str(n_centers-1)}: "))
                        if 0 <= center_id < n_centers:
                            keep_asking_for_input = False
                    except ValueError:
                        warnings.warn(f'Please input a numeric value (integer) between 0 and {str(n_centers-1)}')

                # Build the DataLoadingPlan with the selected dataset type and center id
                data_loading_plan = DataLoadingPlan()
                metadata_dlb = FlambyDatasetMetadataBlock()
                metadata_dlb.metadata = {
                    'flamby_dataset_name': available_flamby_datasets[flamby_dataset_index],
                    'flamby_center_id': center_id
                }
                data_loading_plan[FlambyLoadingBlockTypes.FLAMBY_DATASET_METADATA] = metadata_dlb
            else:
                path = validated_path_input(data_type)

        # if a data loading plan was specified, we now ask for the description
        if interactive and data_loading_plan is not None:
            keep_asking_for_input = True
            while keep_asking_for_input:
                desc = input('Please input a short name/description for your data loading plan:')
                if len(desc) < 4:
                    print('Description must be at least 4 characters long.')
                else:
                    keep_asking_for_input = False
            data_loading_plan.desc = desc

    else:
        # all data have been provided at call
        # check few things

        # transform a string with coma(s) as a string list
        tags = str(tags).split(',')

        name = str(name)
        description = str(description)

        data_type = str(data_type).lower()
        if data_type not in [ 'csv', 'default', 'mednist', 'images' ]:
            data_type = 'default'

        if not os.path.exists(path):
            logger.critical("provided path does not exists: " + path)

    logger.info(f"PATH VALUE {path}")
    # Add database
    try:
        dataset_manager.add_database(name=name,
                                     tags=tags,
                                     data_type=data_type,
                                     description=description,
                                     path=path,
                                     dataset_parameters=dataset_parameters,
                                     data_loading_plan=data_loading_plan)
    except (AssertionError, FedbiomedDatasetManagerError) as e:
        if interactive is True:
            try:
                tkinter.messagebox.showwarning(title='Warning', message=str(e))
            except ModuleNotFoundError:
                warnings.warn(f'[ERROR]: {e}')
        else:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)
    except FedbiomedDatasetError as err:
        warnings.warn(f'[ERROR]: {err} ... Aborting'
                      "\nHint: are you sure you have selected the correct index in Demographic file?")
    print('\nGreat! Take a look at your data:')
    dataset_manager.list_my_data(verbose=True)


def delete_database(interactive: bool = True):
    """Removes one or more dataset from the node's database.

    Does not modify the dataset's files.

    Args:
        interactive:

            - if `True` interactively queries (repeatedly) from the command line
                for a dataset to delete
            - if `False` delete MNIST dataset if it exists in the database
    """
    my_data = dataset_manager.list_my_data(verbose=False)
    if not my_data:
        logger.warning('No dataset to delete')
        return

    if interactive is True:
        options = [d['name'] for d in my_data]
        msg = "Select the dataset to delete:\n"
        msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

    while True:
        try:
            if interactive is True:
                opt_idx = int(input(msg)) - 1
                assert opt_idx in range(len(my_data))

                tags = my_data[opt_idx]['tags']
            else:
                tags = ''
                for ds in my_data:
                    if ds['name'] == 'MNIST':
                        tags = ds['tags']
                        break

            if not tags:
                logger.warning('No matching dataset to delete')
                return
            dataset_manager.remove_database(tags)
            logger.info('Dataset removed. Here your available datasets')
            dataset_manager.list_my_data()
            return
        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def delete_all_database():
    """Deletes all datasets from the node's database.

    Does not modify the dataset's files.
    """
    my_data = dataset_manager.list_my_data(verbose=False)

    if not my_data:
        logger.warning('No dataset to delete')
        return

    for ds in my_data:
        tags = ds['tags']
        dataset_manager.remove_database(tags)
        logger.info('Dataset removed for tags:' + str(tags))

    return
