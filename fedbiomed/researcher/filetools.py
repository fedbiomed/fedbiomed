# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Functions for managing Job/Experiment files.
"""


import os
import re
import shutil
from typing import Tuple, List

from fedbiomed.common.logger import logger


def create_exp_folder(experiments_dir, experimentation_folder: str = None) -> str:
    """Creates a folder for the current experiment (ie the current run of the model).

    Experiment files to keep are stored here: model file, all versions of node parameters,
    all versions of aggregated parameters, breakpoints. The created folder is a
    subdirectory of config.vars[EXPERIMENTS_DIR]

    Args:
        experiments_dir: Base directory for storing experiments files
        experimentation_folder (str, optional): optionaly provide an experimentation
            folder name. This should just contain the name of the folder not a path.
            default; if no folder name is given, generate a `Experiment_x` name where `x-1`
            is the number of experiments already run (`x`=0 for the first experiment)

    Returns:
        Experimentation folder

    Raises:
        PermissionError: cannot create experimentation folder
        OSError: cannot create experimentation folder
        ValueError: bad `experimentation_folder` argument
    """
    if not experimentation_folder:
        # FIXME: improve method robustness (here nb of exp equals nb of files
        # in directory)
        all_files = os.listdir(experiments_dir)
        experimentation_folder = "Experiment_" + str("{:04d}".format(len(all_files)))
    else:
        if os.path.basename(experimentation_folder) != experimentation_folder:
            # experimentation folder cannot be a path
            raise ValueError(f"Bad experimentation folder {experimentation_folder} - " +
                             "it cannot be a path")
    try:
        os.makedirs(os.path.join(experiments_dir, experimentation_folder),
                    exist_ok=True)
    except (PermissionError, OSError) as err:
        logger.error("Can not save experiment files because " +
                     f"{experiments_dir}/{experimentation_folder} " +
                     f"folder could not be created due to {err}")
        raise

    return experimentation_folder


def choose_bkpt_file(
    experiments_dir,
    experimentation_folder: str,
    round_: int = 0
) -> Tuple[str, str]:
    """
    It creates a breakpoint folder and chooses a breakpoint file name for each round.

    Args:
        experiments_dir: Base directory for storing experiments files
        experimentation_folder (str): indicates the experimentation folder name.
            This should just contain the name of the folder not a full path.
        round_: the current number of already run rounds minus one.
            Starts from 0. Defaults to 0.

    Raises:
        PermissionError: cannot create experimentation folder
        OSError: cannot create experimentation folder

    Returns:
        A tuple that contains following instacens
            breakpoint_folder_path: name of the created folder that will contain all data
            for the current round
            breakpoint_file: name of the file that will contain the state of an experiment.
    """

    breakpoint_folder = "breakpoint_" + str("{:04d}".format(round_))
    breakpoint_folder_path = os.path.join(experiments_dir,
                                          experimentation_folder,
                                          breakpoint_folder)

    try:
        os.makedirs(breakpoint_folder_path, exist_ok=True)
    except (PermissionError, OSError) as err:
        logger.error("Can not save breakpoint folder at " +
                     f"{breakpoint_folder_path} due to some error {err}")
        raise
    breakpoint_file = breakpoint_folder + ".json"

    return breakpoint_folder_path, breakpoint_file


def create_unique_link(breakpoint_folder_path: str,
                       link_src_prefix: str,
                       link_src_postfix: str,
                       link_target_path: str) -> str:
    """ Find a non-existing name in `breakpoint_folder_path` and create a symbolic link to a given target name.

    Args:
        breakpoint_folder_path: directory for the source link
        link_src_prefix: beginning of the name for the source link (before unique id)
        link_src_postfix: end of the name for the source link (after unique id)
        link_target_path: target for the symbolic link

    Returns:
        Path of the created link

    Raises:
        PermissionError: cannot create symlink
        OSError: cannot create symlink
        FileExistsError: cannot create symlink
        FileNotFoundError : non-existent directory
    """
    stub = 0
    link_src_path = os.path.join(breakpoint_folder_path,
                                 link_src_prefix + link_src_postfix)

    # Need to ensure unique name for link (e.g. when replaying from non-last breakpoint)
    while os.path.exists(link_src_path) or os.path.islink(link_src_path):
        stub += 1
        link_src_path = os.path.join(breakpoint_folder_path,
                                     link_src_prefix + '_' + str("{:02}".format(stub)) + link_src_postfix)
    try:
        os.symlink(link_target_path, link_src_path)
    except(FileExistsError, PermissionError, OSError, FileNotFoundError) as err:
        logger.error(f"Can not create link to experiment file {link_target_path} " +
                     f"from {link_src_path} due to error {err}")
        raise

    return link_src_path


def create_unique_file_link(breakpoint_folder_path: str, file_path: str) -> str:
    """
    Create a symbolic link in `breakpoint_folder_path` with a non-existing name derived from basename of
    `file_path`. The symbolic link points to the real file targeted by `file_path`

    Args:
        breakpoint_folder_path: directory for the source link
        file_path: path to the target of the link

    Returns:
        Path of the created link

    Raises:
        ValueError: bad name for link source or destination
    """

    try:
        real_file_path = os.path.realpath(file_path)
        real_bkpt_folder_path = os.path.realpath(breakpoint_folder_path)
        if not os.path.isdir(real_bkpt_folder_path) \
                or not os.path.isdir(os.path.dirname(real_file_path)):
            raise ValueError

        # - use relative path for link target for portability
        # - link to the real file, not to a link-to-the-file
        link_target = os.path.relpath(real_file_path, start=real_bkpt_folder_path)
    except ValueError as err:
        mess = 'Saving breakpoint error, ' + \
            f'cannot get relative path to {file_path} from {breakpoint_folder_path}, ' + \
            f'due to error {err}'
        logger.error(mess)
        raise

    # heuristic : assume limited set of characters in filename postfix
    re_src_prefix = re.search("(.+)\.[a-zA-Z]+$",
                              os.path.basename(file_path))
    re_src_postfix = re.search(".+(\.[a-zA-Z]+)$",
                               os.path.basename(file_path))
    if not re_src_prefix or not re_src_postfix:
        error_message = f'Saving breakpoint error, bad filename {file_path} gives ' + \
            f'prefix {re_src_prefix} and postfix {re_src_postfix}'
        logger.error(error_message)
        raise ValueError(error_message)

    link_src_prefix = re_src_prefix.group(1)
    link_src_postfix = re_src_postfix.group(1)

    return create_unique_link(breakpoint_folder_path,
                              link_src_prefix, link_src_postfix,
                              link_target)


def _get_latest_file(pathfile: str,
                     list_name_file: List[str],
                     only_folder: bool = False) -> str:
    """
    Gets the latest file from folder specified in `list_name_file` from the following convention: the more recent folder is
    the file written as `myfile_xx` where `xx` is the higher integer amongst files in `list_name_file`

    Args:
        pathfile: path towards folder on system
        list_name_file: a list containing files
        only_folder: whether to consider only folder names or to consider both  file and folder names.

    Raises:
        FileNotFoundError: triggered if none of the names in folder does not match with naming convention.

    Returns:
        More recent file name given naming convention.
    """

    latest_nb = 0
    latest_folder = None

    for exp_folder in list_name_file:
        exp_match = re.search(r'[0-9]*$',
                              exp_folder)

        # folder name ends with numeric caracters
        if len(exp_folder) != exp_match.span()[0]:
            dir_path = os.path.join(pathfile, exp_folder)
            if not only_folder or os.path.isdir(dir_path):
                f_idx = exp_match.span()[0]
                order = int(exp_folder[f_idx:])
                # found a folder with greater trailing number than current
                # greater trailing number
                if order >= latest_nb:
                    latest_nb = order
                    latest_folder = exp_folder

    if latest_folder is None:
        if len(list_name_file) != 0:
            raise FileNotFoundError(
                "None of those are breakpoints {}".format(", ".join(list_name_file)))
        else:
            raise FileNotFoundError("No files to search for breakpoint")

    return latest_folder


def find_breakpoint_path(experiment_dir: str, breakpoint_folder_path: str | None = None) -> Tuple[str, str]:
    """ Finds breakpoint folder path and file, depending on if user specifies a
    specific breakpoint path (unchanged in this case) or not (try to guess the
    latest breakpoint).

    Args:
        experiments_dir: Base directory for storing experiments files
        breakpoint_folder_path: path towards breakpoint. If None, (default), consider the
            latest breakpoint saved on default path (provided at least one breakpoint
            exists). Defaults to None.

    Returns:
        With length of two that represents respectively:

            - path to breakpoint folder (unchanged if specified by user)
            - breakpoint file.

    Raises:
        FileNotFoundError: triggered either if breakpoint cannot be found, folder is empty or file cannot be parsed
    """

    # First, let's test if folder is a real folder path
    if breakpoint_folder_path is None:
        try:
            # retrieve latest experiment

            # for error message
            latest_exp_folder = experiment_dir + "/NO_FOLDER_FOUND"

            # content of breakpoint folder
            experiment_folders = os.listdir(experiment_dir)

            latest_exp_folder = _get_latest_file(
                experiment_dir,
                experiment_folders,
                only_folder=True)

            latest_exp_folder = os.path.join(experiment_dir,
                                             latest_exp_folder)

            bkpt_folders = os.listdir(latest_exp_folder)

            breakpoint_folder_path = _get_latest_file(
                latest_exp_folder,
                bkpt_folders,
                only_folder=True)
            breakpoint_folder_path = os.path.join(latest_exp_folder,
                                                  breakpoint_folder_path)
        except FileNotFoundError as err:
            logger.error("Cannot find latest breakpoint in " + latest_exp_folder +
                         " Are you sure at least one breakpoint is saved there ? " +
                         " - Error: " + str(err))
            raise
    else:
        if not os.path.isdir(breakpoint_folder_path):
            raise FileNotFoundError(
                f"Breakpoint folder {breakpoint_folder_path} is not a directory")

    # check if folder is a valid breakpoint

    #
    # verify the validity of the breakpoint content
    # TODO: be more robust
    all_breakpoint_materials = os.listdir(breakpoint_folder_path)
    if len(all_breakpoint_materials) == 0:
        raise FileNotFoundError(f'Breakpoint folder {breakpoint_folder_path} is empty !')

    state_file = None
    for breakpoint_material in all_breakpoint_materials:
        # look for the json file containing experiment state
        # (it should be named `breakpoint_xx.json`)
        json_match = re.fullmatch(r'breakpoint_\d*\.json',
                                  breakpoint_material)
        # there should be at most one - TODO: verify
        if json_match is not None:
            logger.debug(f"found json file containing states at\
                {breakpoint_material}")
            state_file = breakpoint_material

    if state_file is None:
        message = "Cannot find JSON file containing " + \
            f"model state at {breakpoint_folder_path}. Aborting"
        logger.error(message)
        raise FileNotFoundError(message)

    return breakpoint_folder_path, state_file


def copy_file(filepath: str, breakpoint_path: str) -> str:
    filename = os.path.dirname(filepath)
    file_copy_path = os.path.join( breakpoint_path, filename)
    shutil.copy2(filepath, file_copy_path )
    return file_copy_path
