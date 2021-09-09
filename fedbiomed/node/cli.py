import os
import time
from multiprocessing import Process

import warnings
import readline
import argparse

import tkinter.filedialog
import tkinter.messagebox


from fedbiomed.node.environ import CLIENT_ID
from fedbiomed.node.data_manager import Data_manager
from fedbiomed.node.node import Node


__intro__ = """

   __         _ _     _                          _        _ _            _
  / _|       | | |   (_)                        | |      | (_)          | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |   ___| |_  ___ _ __ | |_
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` |  / __| | |/ _ \ '_ \| __|
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| | | (__| | |  __/ | | | |_
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_|  \___|_|_|\___|_| |_|\__|
"""

data_manager = Data_manager()

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
    try:
        # root = TK()
        # root.withdraw()
        # root.attributes("-topmost", True)
        if mode == 'file':
            return tkinter.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        else:
            return tkinter.filedialog.askdirectory()

    except ModuleNotFoundError:
        if mode == 'file':
            return input('Insert the path of the CSV file: ')
        else:
            return input('Insert the path of the dataset folder: ')


def validated_path_input(data_type):
    while True:
        try:
            if data_type == 'csv':
                path = pick_with_tkinter(mode='file')
                print(path)
                if not path:
                    print('No file was selected. Exiting...')
                    exit(1)
                assert os.path.isfile(path)
            else:
                path = pick_with_tkinter(mode='dir')
                print(path)
                if not path:
                    print('No directory was selected. Exiting...')
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
        data_manager.add_database(name=name, tags=tags, data_type=data_type,
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

def manage_node():
    print('Launching node...')

    data_manager = Data_manager()
    print('\t - Starting communication channel with network...\n')
    node = Node(data_manager)
    node.start_messaging(block=False)

    print('\t - Starting node manager...\n')
    node.node_manager()

def launch_node():
    #p = Process(target=manage_node, name='node-' + CLIENT_ID, args=(data_manager,))
    p = Process(target=manage_node, name='node-' + CLIENT_ID)
    p.daemon = True
    p.start()

    try:
        print('To stop press Ctrl + C.')
        p.join()
    except KeyboardInterrupt:
        p.terminate()
        while(p.is_alive()):
            print("Terminating process " + str(p.pid))
            time.sleep(1)
        print('Exited with code ' + str(p.exitcode))
        exit()

def delete_database(interactive=True):
    my_data = data_manager.list_my_data(verbose=False)
    if not my_data:
        print('No dataset to delete')
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
                print('No matching dataset to delete')
                return
            data_manager.remove_database(tags)
            print('Dataset removed. Here your available datasets')
            data_manager.list_my_data()
            return
        except (ValueError, IndexError, AssertionError):
            print('Invalid option. Please, try again.')


def launch_cli():

    parser = argparse.ArgumentParser(description=f'{__intro__}:A CLI app for fedbiomed researchers.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-a', '--add', help='Add and configure local dataset (interactive)', action='store_true')
    parser.add_argument('-am', '--add-mnist', help='Add MNIST local dataset (non-interactive)', 
        type=str, nargs='?', const='', metavar='path_mnist', action='store')
    parser.add_argument('-d', '--delete', help='Delete existing local dataset (interactive)', action='store_true')
    parser.add_argument('-dm', '--delete-mnist', help='Delete existing MNIST local dataset (non-interactive)', action='store_true')
    parser.add_argument('-l', '--list', help='List my shared_data', action='store_true')
    parser.add_argument('-s', '--start-node', help='Start fedbiomed node.', action='store_true')
    args = parser.parse_args()

    if not any(args.__dict__.values()):
        parser.print_help()
    else:
        print(__intro__)
        print('\t- 🆔 Your client ID:', CLIENT_ID, '\n')

    if args.add:
        add_database()
    elif args.add_mnist is not None:
        add_database(interactive=False, path=args.add_mnist)
    elif args.list:
        print('Listing your data available...')
        data = data_manager.list_my_data(verbose=True)
        if len(data) == 0:
            print('No data has been set up.')
    elif args.delete:
        delete_database()
    elif args.delete_mnist:
        delete_database(interactive=False)
    elif args.start_node:
        launch_node()


def main():
    try:
        launch_cli()
    except KeyboardInterrupt:
        print('Operation cancelled by user.')

if __name__ == '__main__':
        main()
