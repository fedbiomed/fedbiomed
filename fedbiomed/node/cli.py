import os
import signal
import sys
import time
from multiprocessing import Process

import warnings
import readline
import argparse

import tkinter.filedialog
import tkinter.messagebox
from tkinter import _tkinter

from fedbiomed.node.environ import environ
from fedbiomed.node.data_manager import Data_manager
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

data_manager = Data_manager()
model_manager = ModelManager()

readline.parse_and_bind("tab: complete")


def validated_data_type_input():
    valid_options = ['csv', 'default', 'images']
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


def pick_with_tkinter(mode='file'):
    """
    Opens a tkinter graphical user interface to select dataset

    Args:
        mode (str, optional)
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
        if mode == 'file':
            return input('Insert the path of the CSV file: ')
        else:
            return input('Insert the path of the dataset folder: ')


def validated_path_input(type):
    while True:
        try:
            if type == 'csv':
                path = pick_with_tkinter(mode='file')
                logger.debug(path)
                if not path:
                    logger.critical('No file was selected. Exiting')
                    exit(1)
                assert os.path.isfile(path)

            elif type == 'py': # For registering python model 
                path = pick_with_tkinter(mode='txt')
                logger.debug(path)
                if not path:
                    logger.critical('No python file was selected. Exiting')
                    exit(1)
                assert os.path.isfile(path)
            else:
                path = pick_with_tkinter(mode='dir')
                logger.debug(path)
                if not path:
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


def add_database(interactive=True, path=''):

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

    else:
        name = input('Name of the database: ')

        tags = input('Tags (separate them by comma and no spaces): ')
        tags = tags.replace(' ', '').split(',')

        description = input('Description: ')
        path = validated_path_input(data_type)

    # Add database

    try:
        data_manager.add_database(name=name,
                                  tags=tags,
                                  data_type=data_type,
                                  description=description,
                                  path=path)
    except AssertionError as e:
        if interactive is True:
            try:
                tkinter.messagebox.showwarning(title='Warning', message=str(e))
            except ModuleNotFoundError:
                warnings.warn('[ERROR]: {e}')
        else:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    data_manager.list_my_data(verbose=True)


def node_signal_handler(signum, frame):
    """
    Catch the temination signal then user stops the process
    and send SystemExit(0) to be trapped later
    """
    logger.critical("Node stopped in signal_handler, probably by user decision (Ctrl C)")
    time.sleep(1)
    sys.exit(signum)

def manage_node():
    """
    Instantiates a node and data manager objects. Then, node starts
    messaging with the Network
    """

    try:
        signal.signal(signal.SIGTERM, node_signal_handler)

        logger.info('Launching node...')

        # Register default models and update hashes 
        if environ["MODEL_APPROVE"]:
            # This methods updates hashes if security level has changed
            model_manager.check_hashes_for_registered_models()
            if environ["ALLOW_DEFAULT_MODELS"]:
                logger.info('Loading default models')
                model_manager.register_update_default_models() 
        
        data_manager = Data_manager()
        logger.info('Starting communication channel with network')
        node = Node(data_manager)
        node.start_messaging(block=False)

        logger.info('Starting task manager')
        node.task_manager()  # handling training tasks in queue

    except Exception as e:
        # must send info to the researcher
        logger.critical("Node stopped. Error = " + str(e))

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

def launch_node():
    """
    Launches node in a process. Process ends when user triggers
    a KeyboardInterrupt exception (CTRL+C).
    """

    #p = Process(target=manage_node, name='node-' + environ['NODE_ID'], args=(data_manager,))
    p = Process(target=manage_node, name='node-' + environ['NODE_ID'])
    p.daemon = True
    p.start()

    logger.info("Node started as process with pid = "+ str(p.pid))
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
    my_data = data_manager.list_my_data(verbose=False)
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
            data_manager.remove_database(tags)
            logger.info('Dataset removed. Here your available datasets')
            data_manager.list_my_data()
            return
        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def register_model(interactive: bool = True):

    """ Method for registring model files through CLI """

    print('Welcome to the Fedbiomed CLI data manager')
    name = input('Please enter a model name: ')
    description = input('Please enter a description for the model: ')
    
    # Allow files saved as txt
    path = validated_path_input(type = "txt")

    # Regsiter model 
    try:
        model_manager.register_model(name=name,
                                    description=description,
                                    path=path,
                                    verbose=True)
        
    except AssertionError as e:
        if interactive is True:
            try:
                tkinter.messagebox.showwarning(title='Warning', message=str(e))
            except ModuleNotFoundError:
                warnings.warn('[ERROR]: {e}')
        else:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    model_manager.list_approved_models(verbose=True)

def launch_cli():

    parser = argparse.ArgumentParser(description=f'{__intro__}:A CLI app for fedbiomed researchers.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-a', '--add',
                        help='Add and configure local dataset (interactive)',
                        action='store_true')
    parser.add_argument('-am', '--add-mnist',
                        help='Add MNIST local dataset (non-interactive)',
                        type=str, nargs='?', const='', metavar='path_mnist',
                        action='store')
    parser.add_argument('-d', '--delete',
                        help='Delete existing local dataset (interactive)',
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
    parser.add_argument('-r', '--register-model',
                        help='Start fedbiomed node.',
                        action='store_true')
    parser.add_argument('-lms', '--list-models',
                        help='Start fedbiomed node.',
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
    elif args.list:
        print('Listing your data available')
        data = data_manager.list_my_data(verbose=True)
        if len(data) == 0:
            print('No data has been set up.')
    elif args.delete:
        delete_database()
    elif args.delete_mnist:
        delete_database(interactive=False)
    elif args.register_model:
        register_model()
    elif args.list_models:
        model_manager.list_approved_models(verbose = True)
    elif args.start_node:
        launch_node()


def main():
    try:
        launch_cli()
    except KeyboardInterrupt:
        #print('Operation cancelled by user.')

        # send error message to researche via logger.error()
        logger.critical('Operation cancelled by user.')


if __name__ == '__main__':
    main()
