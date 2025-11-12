# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

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


def pick_with_tkinter(type: str = "csv") -> str:
    """Opens a tkinter graphical user interface to select dataset.

    Args:
        type: type of file to select. Can be `txt` (for .txt files)
            or `csv` (for .csv files)
            Defaults to `csv`.

    Returns:
        The selected path.
    """
    match type:
        case "csv":
            is_file_mode = True
            filetypes = [("CSV files", "*.csv")]
            error_msg_empty = "No file was selected. Exiting"
        case "txt":
            is_file_mode = True
            filetypes = [("Text files", "*.txt")]
            error_msg_empty = "No python file was selected. Exiting"
        case _:
            is_file_mode = False
            filetypes = None
            error_msg_empty = "No directory was selected. Exiting"

    try:
        path = (
            filedialog.askopenfilename(filetypes=filetypes)
            if is_file_mode
            else filedialog.askdirectory()
        )

        # Window was closed or cancelled
        if not path:
            logger.critical(error_msg_empty)
            exit(1)

        logger.debug(path)

    except (RuntimeError, TclError):
        path = None
        error_msg = "[ERROR] GUI failed. Falling back to CLI"
        messagebox.showerror(title="Error", message=error_msg)

    return path


def validated_path_input(type: str) -> str:
    """Picks path to use from user input in GUI or command line.

    Args:
        type: keyword for the kind of object pointed by the path.

    Returns:
        The selected path.
    """
    if filedialog is None:
        warnings.warn("[WARNING] GUI not available. Falling back to CLI", stacklevel=1)

    # Try GUI first if available
    path = None if filedialog is None else pick_with_tkinter(type=type)

    # Determine if we are in file mode
    is_file_mode = type in ("csv", "txt")

    while True:
        # CLI fallback
        if path is None:
            prompt = f"Insert the path of the {'file' if is_file_mode else 'folder'}: "
            path = input(prompt)

            if not path.strip():
                warnings.warn("[ERROR] Path cannot be empty", stacklevel=1)
                continue

            path_obj = Path(path).expanduser()
            valid_path = path_obj.is_file() if is_file_mode else path_obj.is_dir()
            if not valid_path:
                warnings.warn("[ERROR] Please enter a valid path", stacklevel=1)
                continue
            path = str(path_obj)

        return path
