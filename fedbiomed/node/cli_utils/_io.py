# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import tkinter.filedialog
import tkinter.messagebox
from tkinter import _tkinter
import warnings

from fedbiomed.common.logger import logger


def validated_data_type_input() -> str:
    """Picks data type to use from user input on command line.
    Returns:
        A string keyword for one of the possible data type
            ('csv', 'default', 'mednist', 'images', 'medical-folder', 'flamby').
    """
    valid_options = ['csv', 'default', 'mednist', 'images', 'medical-folder', 'flamby']
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

    except (ModuleNotFoundError, RuntimeError, _tkinter.TclError):
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
                    logger.critical('No file was selected. Exiting')
                    exit(1)
                assert os.path.isfile(path)

            elif type == 'txt':  # for registering python model
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
