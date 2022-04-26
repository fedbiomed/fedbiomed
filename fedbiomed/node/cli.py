"""
Command line user interface for the node component
"""

import json
import os
import signal
import sys
import time
from multiprocessing import Process
from typing import Union
from types import FrameType

import warnings
import readline
import argparse

import tkinter.filedialog
import tkinter.messagebox
from tkinter import _tkinter

from fedbiomed.common.constants  import ModelTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedDatasetManagerError

from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.environ import environ
from fedbiomed.node.model_manager import ModelManager
from fedbiomed.node.node import Node

from fedbiomed.common.logger import logger


#
# print(pyfiglet.Figlet("doom").renderText(' fedbiomed node'))
#
__intro__ = """

   __         _ _     _                          _                   _
  / _|       | | |   (_)                        | |                 | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |  _ __   ___   __| | ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _ \ / _` |/ _ \\
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | | | | (_) | (_| |  __/
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_| |_| |_|\___/ \__,_|\___|


"""

# this may be changed on command line or in the config_node.ini
logger.setLevel("DEBUG")

dataset_manager = DatasetManager()
model_manager = ModelManager()

readline.parse_and_bind("tab: complete")


def validated_data_type_input() -> str:
    """Picks data type to use from user input on command line.

    Returns:
        A string keyword for one of the possible data type
            ('csv', 'default', 'mednist', 'images').
    """
    valid_options = ['csv', 'default', 'mednist', 'images']
    valid_options = {i: val for i, val in enumerate(valid_options, 1)}

    msg = "Please select the data type that you're configuring:\n"
    msg += "\n".join([f"\t{i}) {val}" for i, val in valid_options.items()])
    msg += "\nselect: "

    while True:
        try:
            t = int(input(msg))
            assert t in valid_options.keys()
            break
        except Exception:
            warnings.warn('\n[ERROR] Please, enter a valid option')

    return valid_options[t]


def pick_with_tkinter(mode: str = 'file') -> str:
    """Opens a tkinter graphical user interface to select dataset.

    Args:
        mode: type of file to select. Can be `txt` (for .txt files)
            or `file` (for .csv files)
            Defaults to `file`.

    Returns:
        The selected path.
    """
    try:
        # root = TK()
        # root.withdraw()
        # root.attributes("-topmost", True)
        if mode == 'file':
            return tkinter.filedialog.askopenfilename(
                filetypes=[
                    ("CSV files",
                     "*.csv")
                ]
            )
        elif mode == 'txt':
            return tkinter.filedialog.askopenfilename(
                filetypes=[
                    ("Text files",
                     "*.txt")
                ]
            )
        else:
            return tkinter.filedialog.askdirectory()

    except (ModuleNotFoundError, _tkinter.TclError):
        # handling case where tkinter package cannot be found on system
        # or if tkinter crashes
        if mode == 'file' or mode == 'txt':
            return input('Insert the path of the file: ')
        else:
            return input('Insert the path of the folder: ')


def validated_path_input(type: str) -> str:
    """Picks path to use from user input in GUI or command line.

    Args:
        type: keyword for the kind of object pointed by the path.

    Returns:
        The selected path.
    """
    while True:
        try:
            if type == 'csv':
                path = pick_with_tkinter(mode='file')
                logger.debug(path)
                if not path:
                    # node is not in computation mode, MQTT message cannot be sent
                    logger.critical('No file was selected. Exiting')
                    exit(1)
                assert os.path.isfile(path)

            elif type == 'txt':  # for registering python model
                path = pick_with_tkinter(mode='txt')
                logger.debug(path)
                if not path:
                    # node is not in computation mode, MQTT message cannot be sent
                    logger.critical('No python file was selected. Exiting')
                    exit(1)
                assert os.path.isfile(path)
            else:
                path = pick_with_tkinter(mode='dir')
                logger.debug(path)
                if not path:
                    # node is not in computation mode, MQTT message cannot be sent
                    logger.critical('No directory was selected. Exiting')
                    exit(1)
                assert os.path.isdir(path)
            break
        except Exception:
            error_msg = '[ERROR] Invalid path. Please enter a valid path.'
            try:
                tkinter.messagebox.showerror(title='Error', message=error_msg)
            except ModuleNotFoundError:
                warnings.warn(error_msg)

    return path


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

    # if all args are provided, just try to load the data
    # if not, ask the user more informations
    if interactive or \
       path is None or \
       name is None or \
       tags is None or \
       description is None or \
       data_type is None :


        print('Welcome to the Fedbiomed CLI data manager')

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

    # Add database
    try:
        dataset_manager.add_database(name=name,
                                     tags=tags,
                                     data_type=data_type,
                                     description=description,
                                     path=path)
    except (AssertionError, FedbiomedDatasetManagerError) as e:
        if interactive is True:
            try:
                tkinter.messagebox.showwarning(title='Warning', message=str(e))
            except ModuleNotFoundError:
                warnings.warn(f'[ERROR]: {e}')
        else:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    dataset_manager.list_my_data(verbose=True)


def node_signal_handler(signum: int, frame: Union[FrameType, None]):
    """Signal handler that terminates the process.

    Args:
        signum: Signal number received.
        frame: Frame object received. Currently unused

    Raises:
       SystemExit: Always raised.
    """

    # get the (running) Node object
    global node

    if node:
        node.send_error(ErrorNumbers.FB312)
    else:
        logger.error("Cannot send error message to researcher (node not initialized yet)")
    logger.critical("Node stopped in signal_handler, probably by user decision (Ctrl C)")
    time.sleep(1)
    sys.exit(signum)


def manage_node(node_args: Union[dict, None] = None):
    """Runs the node component and blocks until the node terminates.

    Intended to be launched by the node in a separate process/thread.

    Instantiates `Node` and `DatasetManager` object, start exchaning 
    messages with the researcher via the `Node`, passes control to the `Node`.

    Args:
        node_args: command line arguments for node.
            See `Round()` for details.
    """

    global node

    try:
        signal.signal(signal.SIGTERM, node_signal_handler)

        logger.info('Launching node...')

        # Register default models and update hashes
        if environ["MODEL_APPROVAL"]:
            # This methods updates hashes if hashing algorithm has changed
            model_manager.check_hashes_for_registered_models()
            if environ["ALLOW_DEFAULT_MODELS"]:
                logger.info('Loading default models')
                model_manager.register_update_default_models()
        else:
            logger.warning('Model approval for train request is not activated. ' +
                           'This might cause security problems. Please, consider to enable model approval.')

        dataset_manager = DatasetManager()
        logger.info('Starting communication channel with network')
        node = Node(dataset_manager = dataset_manager,
                    model_manager = model_manager,
                    node_args=node_args)
        node.start_messaging(block=False)

        logger.info('Starting task manager')
        node.task_manager()  # handling training tasks in queue

    except FedbiomedError:
        logger.critical("Node stopped.")
        # we may add extra information for the user depending on the error

    except Exception as e:
        # must send info to the researcher (no mqqt should be handled by the previous FedbiomedError)
        node.send_error(ErrorNumbers.FB300, extra_msg = "Error = " + str(e))
        logger.critical("Node stopped.")

    finally:
        # this is triggered by the signal.SIGTERM handler SystemExit(0)
        #
        # cleaning staff should be done here
        pass


    # finally:
    #     # must send info to the researcher (as critical ?)
    #     logger.critical("(CRIT)Node stopped, probably by user decision (Ctrl C)")
    #     time.sleep(1)
    #     logger.exception("Reason:")
    #     time.sleep(1)

def launch_node(node_args: Union[dict, None] = None):
    """Launches a node in a separate process.

    Process ends when user triggers a KeyboardInterrupt exception (CTRL+C).

    Args:
        node_args: Command line arguments for node
            See `Round()` for details.
    """

    p = Process(target=manage_node, name='node-' + environ['NODE_ID'], args=(node_args,))
    p.daemon = True
    p.start()

    logger.info("Node started as process with pid = " + str(p.pid))
    try:
        print('To stop press Ctrl + C.')
        p.join()
    except KeyboardInterrupt:
        p.terminate()

        # give time to the node to send a MQTT message
        time.sleep(1)
        while(p.is_alive()):
            logger.info("Terminating process id =" + str(p.pid))
            time.sleep(1)

        # (above) p.exitcode returns None if not finished yet
        logger.info('Exited with code ' + str(p.exitcode))

        exit()


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
                assert opt_idx >= 0

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


def register_model():
    """Registers an authorized model in the database interactively through the CLI.

    Does not modify model file.
    """

    print('Welcome to the Fedbiomed CLI data manager')
    name = input('Please enter a model name: ')
    description = input('Please enter a description for the model: ')

    # Allow files saved as txt
    path = validated_path_input(type = "txt")

    # Register model
    try:
        model_manager.register_model(name = name,
                                     description = description,
                                     path = path)

    except AssertionError as e:
        try:
            tkinter.messagebox.showwarning(title='Warning', message=str(e))
        except ModuleNotFoundError:
            warnings.warn('[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    model_manager.list_approved_models(verbose=True)


def update_model():
    """Updates an authorized model in the database interactively through the CLI.

    Does not modify model file.

    User can either choose different model file (different path)
    to update model or same model file.
    """
    models = model_manager.list_approved_models(verbose=False)

    # Select only registered model to update
    models = [ m for m in models  if m['model_type'] == ModelTypes.REGISTERED.value]
    if not models:
        logger.warning('No registered models has been found to update')
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] for m in models]
    msg = "Select the model to update:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            # Get the selection
            opt_idx = int(input(msg)) - 1
            assert opt_idx >= 0
            model_id = models[opt_idx]['model_id']

            if not model_id:
                logger.warning('No matching model to update')
                return

            # Get the new file or same file.  User can provide same model file
            # with updated content or new model file.
            path = validated_path_input(type = "txt")

            # Update model through model manager
            model_manager.update_model(model_id, path)

            logger.info('Model has been updated. Here all your models')
            model_manager.list_approved_models(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def delete_model():
    """Deletes an authorized model in the database interactively from the CLI.

    Does not modify or delete model file.

    Deletes only registered models. For default models, files
    should be removed directly from the file system.
    """

    models = model_manager.list_approved_models(verbose=False)
    models = [ m for m in models  if m['model_type'] == ModelTypes.REGISTERED.value]
    if not models:
        logger.warning('No models to delete')
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] for m in models]
    msg = "Select the model to delete:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            opt_idx = int(input(msg)) - 1
            assert opt_idx >= 0
            model_id = models[opt_idx]['model_id']

            if not model_id:
                logger.warning('No matching model to delete')
                return
            # Delete model
            model_manager.delete_model(model_id)
            logger.info('Model has been removed. Here your other models')
            model_manager.list_approved_models(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def launch_cli():
    """Parses command line input for the node component and launches node accordingly.
    """

    parser = argparse.ArgumentParser(description=f'{__intro__}:A CLI app for fedbiomed researchers.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-a', '--add',
                        help='Add and configure local dataset (interactive)',
                        action='store_true')
    parser.add_argument('-am', '--add-mnist',
                        help='Add MNIST local dataset (non-interactive)',
                        type=str, nargs='?', const='', metavar='path_mnist',
                        action='store')
    # this option provides a json file describing the data to add
    parser.add_argument('-adff', '--add-dataset-from-file',
                        help='Add a local dataset described by json file (non-interactive)',
                        type=str,
                        action='store')
    parser.add_argument('-d', '--delete',
                        help='Delete existing local dataset (interactive)',
                        action='store_true')
    parser.add_argument('-da', '--delete-all',
                        help='Delete all existing local datasets (non interactive)',
                        action='store_true')
    parser.add_argument('-dm', '--delete-mnist',
                        help='Delete existing MNIST local dataset (non-interactive)',
                        action='store_true')
    parser.add_argument('-l', '--list',
                        help='List my shared_data',
                        action='store_true')
    parser.add_argument('-s', '--start-node',
                        help='Start fedbiomed node.',
                        action='store_true')
    parser.add_argument('-rml', '--register-model',
                        help='Approve new model files.',
                        action='store_true')
    parser.add_argument('-uml', '--update-model',
                        help='Update model file.',
                        action='store_true')
    parser.add_argument('-dml', '--delete-model',
                        help='Deletes models from DB',
                        action='store_true')
    parser.add_argument('-lms', '--list-models',
                        help='Start fedbiomed node.',
                        action='store_true')
    parser.add_argument('-g', '--gpu',
                        help='Use of a GPU device, if any available (default: dont use GPU)',
                        action='store_true')
    parser.add_argument('-gn', '--gpu-num',
                        help='Use GPU device with the specified number instead of default device, if available',
                        type=int,
                        action='store')
    parser.add_argument('-go', '--gpu-only',
                        help='Force use of a GPU device, if any available, even if researcher doesnt ' +
                        'request it (default: dont use GPU)',
                        action='store_true')
    args = parser.parse_args()

    if not any(args.__dict__.values()):
        parser.print_help()
    else:
        print(__intro__)
        print('\t- 🆔 Your node ID:', environ['NODE_ID'], '\n')

    if args.add:
        add_database()
    elif args.add_mnist is not None:
        add_database(interactive=False, path=args.add_mnist)
    elif args.add_dataset_from_file is not None:
        print("Dataset description file provided: adding these data")
        try:
            with open(args.add_dataset_from_file) as json_file:
                data = json.load(json_file)
        except:
            logger.critical("cannot read dataset json file: " + args.add_dataset_from_file)
            sys.exit(-1)

        # verify that json file is complete
        for k in [ "path", "data_type", "description", "tags", "name"]:
            if k not in data:
                logger.critical("dataset json file corrupted: " + args.add_dataset_from_file )

        # dataset path can be defined:
        # - as an absolute path -> take it as it is
        # - as a relative path  -> add the ROOT_DIR in front of it
        # - using an OS environment variable -> transform it
        #
        elements = data["path"].split(os.path.sep)
        if elements[0].startswith("$") :
            # expand OS environment variable
            var = elements[0][1:]
            if var in os.environ:
                var = os.environ[var]
                elements[0] = var
            else:
                logger.info("Unknown env var: " + var)
                elements[0] = ""
        elif elements[0]:
            # p is relative (does not start with /)
            # prepend with topdir
            elements = [ environ["ROOT_DIR"] ] + elements

        # rebuild the path with these (eventually) new elements
        data["path"] = os.path.join(os.path.sep, *elements)

        # add the dataset to local database (not interactive)
        add_database(interactive=False,
                     path        = data["path"],
                     data_type   = data["data_type"],
                     description = data["description"],
                     tags        = data["tags"],
                     name        = data["name"]
                     )

    elif args.list:
        print('Listing your data available')
        data = dataset_manager.list_my_data(verbose=True)
        if len(data) == 0:
            print('No data has been set up.')
    elif args.delete:
        delete_database()
    elif args.delete_all:
        delete_all_database()
    elif args.delete_mnist:
        delete_database(interactive=False)
    elif args.register_model:
        register_model()
    elif args.update_model:
        update_model()
    elif args.delete_model:
        delete_model()
    elif args.list_models:
        model_manager.list_approved_models(verbose = True)
    elif args.start_node:
        # convert to node arguments structure format expected in Round()
        node_args = {
            'gpu': (args.gpu_num is not None) or (args.gpu is True) or (args.gpu_only is True),
            'gpu_num': args.gpu_num,
            'gpu_only': (args.gpu_only is True)
        }
        launch_node(node_args)


def main():
    """Entry point for the node.
    """
    try:
        launch_cli()
    except KeyboardInterrupt:
        # send error message to researche via logger.error()
        logger.critical('Operation cancelled by user.')


if __name__ == '__main__':
    main()
