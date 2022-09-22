import os
import tkinter.messagebox
import warnings

from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedDatasetManagerError
from fedbiomed.common.logger import logger
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.cli_utils._io import validated_data_type_input, validated_path_input
from fedbiomed.node.cli_utils._medical_folder_dataset import add_medical_folder_dataset_from_cli


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
            data_type is None:

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
            else:
                path = validated_path_input(data_type)

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

    if interactive and data_loading_plan is not None:
        print(f'The {data_loading_plan} will be saved.')
        data_loading_plan.name = input('Optionally input a name to help you identify the data loading plan:\n')

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
