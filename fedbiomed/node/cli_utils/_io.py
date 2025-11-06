# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from pathlib import Path

from fedbiomed.common.logger import logger

from ._tkinter_utils import TclError, filedialog, messagebox


def validated_data_type_input() -> str:
    """Picks data type to use from user input on command line.
    Returns:
        A string keyword for one of the possible data type
            ('csv', 'default', 'mednist', 'images', 'medical-folder', 'custom').
    """
    valid_options = [
        "csv",
        "default",
        "mednist",
        "images",
        "medical-folder",
        "custom",
    ]
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
            warnings.warn("\n[ERROR] Please, enter a valid option", stacklevel=1)

    return valid_options[t]


def pick_with_tkinter(mode: str = "file") -> str:
    """Opens a tkinter graphical user interface to select dataset.

    Args:
        mode: type of file to select. Can be `txt` (for .txt files)
            or `file` (for .csv files)
            Defaults to `file`.

    Returns:
        The selected path.
    """
    # Try GUI first if available
    if filedialog is not None:
        try:
            if mode == "file":
                return filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            elif mode == "txt":
                return filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            else:
                return filedialog.askdirectory()

        except (RuntimeError, TclError):
            # GUI failed, fall back to CLI input
            pass

    # Fallback to CLI input with validation
    is_file_mode = mode in ("file", "txt")
    prompt = f"Insert the path of the {'file' if is_file_mode else 'folder'}: "

    while True:
        path = input(prompt)
        if not path.strip():
            warnings.warn("[ERROR] Path cannot be empty", stacklevel=1)
            continue

        path_obj = Path(path).expanduser()

        if is_file_mode and path_obj.is_file():
            return str(path_obj)
        elif not is_file_mode and path_obj.is_dir():
            return str(path_obj)
        else:
            warnings.warn("[ERROR] Please enter a valid path", stacklevel=1)


def validated_path_input(type: str) -> str:
    """Picks path to use from user input in GUI or command line.

    Args:
        type: keyword for the kind of object pointed by the path.

    Returns:
        The selected path.
    """
    while True:
        try:
            if type == "csv":
                path = pick_with_tkinter(mode="file")
                logger.debug(path)
                if not path:
                    logger.critical("No file was selected. Exiting")
                    exit(1)
                assert os.path.isfile(path)

            elif type == "txt":  # for registering python model
                path = pick_with_tkinter(mode="txt")
                logger.debug(path)
                if not path:
                    logger.critical("No python file was selected. Exiting")
                    exit(1)
                assert os.path.isfile(path)
            else:
                path = pick_with_tkinter(mode="dir")
                logger.debug(path)
                if not path:
                    logger.critical("No directory was selected. Exiting")
                    exit(1)
                assert os.path.isdir(path)
            break
        except Exception:
            error_msg = "[ERROR] Invalid path. Please enter a valid path."
            if messagebox is not None:
                messagebox.showerror(title="Error", message=error_msg)
            else:
                warnings.warn(error_msg, stacklevel=1)

    return path
