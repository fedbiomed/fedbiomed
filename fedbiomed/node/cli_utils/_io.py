# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from fedbiomed.common.logger import logger

from ._tkinter_utils import TclError, filedialog, messagebox


def validated_data_type_input() -> str:
    """Picks data type to use from user input on command line.
    Returns:
        A string keyword for one of the possible data type
            ('csv', 'default', 'mednist', 'images', 'medical-folder', 'custom').
    """
    _types = ["csv", "default", "mednist", "images", "medical-folder", "custom"]
    valid_options = dict(enumerate(_types, 1))

    msg = "Please select the data type that you're configuring:\n"
    msg += "\n".join([f"\t{i}) {val}" for i, val in valid_options.items()])
    msg += "\nselect: "

    while True:
        try:
            t = int(input(msg))
            assert t in valid_options.keys()
            break
        except Exception:
            print(
                "Invalid option. Please enter a number between 1 and",
                len(valid_options),
            )

    return valid_options[t]


def pick_with_tkinter(type: str = "csv") -> str | None:
    """Opens a tkinter dialog to pick a file or directory.

    Args:
        type: `'csv'` or `'txt'` opens a file picker; any other value opens a directory picker.

    Returns:
        The selected path, or None if cancelled or the GUI failed.
    """
    if filedialog is None or messagebox is None:
        return None

    _config = {
        "csv": ([("CSV files", "*.csv")], "No file was selected."),
        "txt": ([("Text files", "*.txt")], "No python file was selected."),
    }
    filetypes, empty_msg = _config.get(type, (None, "No directory was selected."))
    is_file_mode = filetypes is not None

    try:
        path = (
            filedialog.askopenfilename(filetypes=filetypes)
            if is_file_mode
            else filedialog.askdirectory()
        )
        if not path:
            logger.warning(empty_msg)
            return None
        return path
    except (RuntimeError, TclError) as e:
        error_msg = f"GUI failed: {e}. Falling back to CLI."
        logger.warning(error_msg)
        try:
            messagebox.showerror(title="Error", message=error_msg)
        except (RuntimeError, TclError):
            pass  # messagebox also unavailable, already logged above
        return None


def _prompt_path_cli(type: str) -> str:
    """Prompts the user for a valid path via CLI only.

    Args:
        type: `'csv'` or `'txt'` for a file; any other value for a directory.

    Returns:
        The selected, validated path.
    """
    is_file_mode = type in ("csv", "txt")
    prompt = f"Insert the path of the {'file' if is_file_mode else 'folder'}: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Path cannot be empty.")
            continue
        path_obj = Path(raw).expanduser()
        if not (path_obj.is_file() if is_file_mode else path_obj.is_dir()):
            print("Please enter a valid path.")
            continue
        return str(path_obj)


def validated_path_input(type: str) -> str:
    """Returns a validated path, trying the GUI first then falling back to CLI.

    Args:
        type: `'csv'` or `'txt'` for a file picker; any other value for a directory picker.

    Returns:
        The selected, validated path.
    """
    if filedialog is not None:
        path = pick_with_tkinter(type=type)
        if path is not None:
            return path
        logger.warning("GUI failed. Falling back to CLI.")
    else:
        logger.warning("GUI not available. Falling back to CLI.")

    return _prompt_path_cli(type)
