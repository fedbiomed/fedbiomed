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
    DB_PREFIX,
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

    try:
        config.read(config_path)
    except Exception:
        raise FedbiomedError(
            f"Can not read config file. Please make sure it is existing or it has valid format. "
            f"{config_path}"
        )

    return config


def get_component_certificate_from_config(config_path: str) -> Dict[str, str]:
    """Gets component certificate, id and component type by given config file path.

    Args:
        config_path: Path where config file is located.

    Returns:
        Certificate object that contains  component type as `component`, party id `id`, public key content
            (not path)  as `certificate`

    Raises:
        FedbiomedError:
            - If config file does not contain `node_id` or `researcher_id` under `default` section.
            - If config file does not contain `public_key` under `ssl` section.
            - If certificate file is not found or not readable
    """

    config = get_component_config(config_path)
    component_id = config.get("default", "id")
    component_type = config.get("default", "component")

    certificate = config.get("certificate", "public_key")
    certificate_path = os.path.join(os.path.dirname(config_path), certificate)

    if not os.path.isfile(certificate_path):
        raise FedbiomedError(
            f"The certificate for component '{component_id}' not found in {certificate_path}"
        )

    certificate = read_file(certificate_path)

    return {
        "party_id": component_id,
        "certificate": certificate,
        "component": component_type,
    }


def get_all_existing_config_files():
    """Gets all existing config files from Fed-BioMed `etc` directory"""
    etc = os.path.join(ROOT_DIR, CONFIG_FOLDER_NAME, "")
    return [file for file in glob.glob(f"{etc}*.ini")]


def get_all_existing_certificates() -> List[Dict[str, str]]:
    """Gets all existing certificates from Fed-BioMed `etc` directory.

    This method parse all available configs in `etc` directory.

    Returns:
        List of certificate objects that contain  component type as `component`,
            party id `id`, public key content (not path)  as `certificate`.
    """

    config_files = get_all_existing_config_files()

    certificates = []
    for config in config_files:
        certificates.append(get_component_certificate_from_config(config))

    return certificates


def get_existing_component_db_names():
    """Gets DB_PATHs of all existing components in Fed-BioMed root"""

    config_files = get_all_existing_config_files()
    db_names = {}

    for _config in config_files:
        config = get_component_config(_config)
        component_id = config["default"]["id"]

        db_name = f"{DB_PREFIX}{component_id}"
        db_names = {**db_names, component_id: db_name}

    return db_names


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
