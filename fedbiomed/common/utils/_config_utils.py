# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import configparser

from typing import List, Dict

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.constants import DB_PREFIX


def get_fedbiomed_root() -> str:
    """Gets fedbiomed root.

    Returns:
        Absolute path of Fed-BioMed root directory
    """

    return os.path.abspath(os.path.join(__file__, '..', "..", "..", ".."))


def get_component_config(
        config_path: str
) -> configparser.ConfigParser:
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
    except Exception as e:
        raise FedbiomedError(f"Can not read config file. Please make sure it is existing or it has valid format. "
                             f"{config_path}")

    return config


def get_component_certificate_from_config(
        config_path: str
) -> Dict[str, str]:
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

    ip = config.get("mpspdz", "mpspdz_ip")
    port = config.get("mpspdz", "mpspdz_port")
    certificate_path = os.path.join(os.path.dirname(config_path), config.get("mpspdz", "public_key"))

    if not os.path.isfile(certificate_path):
        raise FedbiomedError(f"The certificate for component '{component_id}' not found in {certificate_path}")

    try:
        with open(certificate_path, 'r') as file:
            certificate = file.read()
    except Exception as e:
        raise FedbiomedError(f"Error while reading certificate -> {certificate_path}. Error: {e}")

    return {
        "party_id": component_id,
        "certificate": certificate,
        "ip": ip,
        "port": port,
        "component": component_type
    }


def get_all_existing_config_files():
    """Gets all existing config files from Fed-BioMed `etc` directory"""
    etc = os.path.join(get_fedbiomed_root(), 'etc', '')
    return [file for file in glob.glob(f"{etc}*.ini")]


def get_all_existing_certificates() -> List[Dict[str, str]]:
    """Gets all existing certificates from Fed-BioMed `etc` directory.

    This method parse all available configs in `etc` directory.

    Returns:
        List of certificate objects that contain  component type as `component`, party id `id`, public key content
        (not path)  as `certificate`.
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
        component_id = config['default']['id']

        db_name = f"{DB_PREFIX}{component_id}"
        db_names = {**db_names, component_id: db_name}

    return db_names
