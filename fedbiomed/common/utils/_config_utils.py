# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
import configparser
import glob
import os
import site
import sysconfig
from typing import Dict, List

from fedbiomed.common.constants import (
    CACHE_FOLDER_NAME,
    CONFIG_FOLDER_NAME,
    TMP_FOLDER_NAME,
    VAR_FOLDER_NAME,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._utils import read_file


def _get_fedbiomed_root() -> str:
    """Gets fedbiomed root.

    Returns:
        Absolute path of Fed-BioMed root directory
    """

    root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    if "envs" in os.listdir(root):
        return root

    return os.path.abspath(os.path.join(root, ".."))


def _get_shared_dir():
    """Gets data directory where Fed-BioMed static package content is saved"""

    fedbiomed_data_sys = os.path.join(sysconfig.get_path("data"), "share", "fedbiomed")
    fedbiomed_data_user_base = os.path.join(str(site.USER_BASE), "share", "fedbiomed")

    if os.path.isdir(fedbiomed_data_sys):
        return fedbiomed_data_sys

    if not os.path.isdir(fedbiomed_data_user_base):
        raise FedbiomedError(
            f"Can not find fedbiomed package data in {fedbiomed_data_sys} "
            f"or {fedbiomed_data_user_base}"
        )

    return fedbiomed_data_user_base


# Main directories definition
ROOT_DIR = _get_fedbiomed_root()
SHARE_DIR = _get_shared_dir()


def get_component_config(config_path: str) -> configparser.ConfigParser:
    """Gets config object from given config path.

    Args:
        config_path: The path where config file is stored.

    Returns:
        Configuration object.

    Raises:
        FedbiomedError: If config file is not readable or not existing.
    """
    config = configparser.ConfigParser()

    if not os.path.isfile(config_path):
        raise FedbiomedError(f"Config file is not existing. {config_path}")

    try:
        config.read(config_path)
    except Exception as e:
        raise FedbiomedError(
            f"Can not read config file. Please make sure it has valid format. "
            f"{config_path}"
        ) from e

    return config


def get_component_certificate_from_config(config_path: str) -> Dict[str, str]:
    """Gets component certificate, id and component type by given config file path.

    Args:
        config_path: Path where config file is located.

    Returns:
        Certificate object that contains component type as `component_type`
            (uppercase), component id `id`, public key content (not path) as
            `certificate`

    Raises:
        FedbiomedError:
            - If config file does not contain `id` or `component` under `default` section.
            - If config file does not contain `public_key` under `certificate` section.
            - If certificate file is not found or not readable
    """

    config = get_component_config(config_path)

    try:
        component_id = config.get("default", "id")
        component_type = config.get("default", "component").upper()
        certificate = config.get("certificate", "public_key")
    except configparser.Error as e:
        raise FedbiomedError(
            f"Config file is missing component or certificate declarations. "
            f"{config_path}: {e}"
        ) from e

    certificate_path = os.path.join(os.path.dirname(config_path), certificate)

    if not os.path.isfile(certificate_path):
        raise FedbiomedError(
            f"The certificate for component '{component_id}' not found in {certificate_path}"
        )

    certificate = read_file(certificate_path)

    return {
        "component_id": component_id,
        "certificate": certificate,
        "component_type": component_type,
    }


def get_all_existing_config_files(path: str) -> List[str]:
    """Gets config files of the components directly under the given directory.

    Only the first level is inspected.

    Args:
        path: Directory the components are located in.

    Returns:
        Paths of the configuration files found, one per component.
    """
    config_files = []
    for entry in sorted(glob.glob(os.path.join(path, "*"))):
        # Every component root keeps its configuration under this name
        config_file = os.path.join(entry, CONFIG_FOLDER_NAME, "config.ini")
        if os.path.isfile(config_file):
            config_files.append(config_file)

    return config_files


def get_all_existing_certificates(path: str) -> List[Dict[str, str]]:
    """Gets certificates of the components directly under the given directory.

    Args:
        path: Directory the components are located in.

    Returns:
        List of certificate objects that contain  component type as `component_type`,
            component id `id`, public key content (not path)  as `certificate`.
    """

    config_files = get_all_existing_config_files(path)

    certificates = []
    for config in config_files:
        certificates.append(get_component_certificate_from_config(config))

    return certificates


def get_existing_component_db_paths(path: str) -> Dict[str, str]:
    """Gets database paths of the components directly under the given directory.

    Taken from each component's own configuration, so it is the database it
    reads at runtime.

    Args:
        path: Directory the components are located in.

    Returns:
        Absolute database path of each component, by component id.
    """

    config_files = get_all_existing_config_files(path)
    db_paths = {}

    for _config in config_files:
        config = get_component_config(_config)
        component_id = config["default"]["id"]

        db_paths[component_id] = os.path.abspath(
            os.path.join(os.path.dirname(_config), config["default"]["db"])
        )

    return db_paths


def create_fedbiomed_setup_folders(root: str):
    """Creates folders reequired by Fed-BioMed component setup

    Args:
        root: Root directory of Fed-BioMed component setup
    """

    etc_config_dir = os.path.join(root, CONFIG_FOLDER_NAME)
    var_dir = os.path.join(root, VAR_FOLDER_NAME)
    cache_dir = os.path.join(var_dir, CACHE_FOLDER_NAME)
    tmp_dir = os.path.join(var_dir, TMP_FOLDER_NAME)

    for dir_ in [etc_config_dir, var_dir, cache_dir, tmp_dir]:
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

    return etc_config_dir, var_dir, cache_dir, tmp_dir
