import glob
import os
import configparser

from typing import List, Dict, Tuple

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.constants import ComponentType

DB_PREFIX = "db_"


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


def get_component_from_config(
        config_path: str
) -> Tuple[str, str, configparser.ConfigParser]:
    """Gets component id, type and config object from config path

    Args:
        config_path: The path of config file

    Returns:
        component_id: The ID of the component.
        component_type:
        config:
    """

    config = get_component_config(config_path)
    if config.has_option("default", "node_id"):
        component_id = config["default"]["node_id"]
        component = ComponentType.NODE.name
    elif config.has_option("default", "researcher_id"):
        component_id = config["default"]["researcher_id"]
        component = ComponentType.RESEARCHER.name
    else:
        raise FedbiomedError(f"Component id is not existing in {config_path}")

    return component_id, component, config


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

    component_id, component, config = get_component_from_config(config_path)

    config = get_component_config(config_path)

    if not config.has_option("ssl", "public_key"):
        raise FedbiomedError(f"Component {component_id} does not have certificate section in the config.")

    certificate_path = config.get("ssl", "public_key")

    if not os.path.isfile(certificate_path):
        raise FedbiomedError(f"The certificate for component '{component_id}' not found in {certificate_path}")

    try:
        with open(certificate_path, 'r') as file:
            certificate = file.read()
    except Exception as e:
        raise FedbiomedError(f"Error while reading certificate -> {certificate_path}. Error: {e}")

    return {"party_id": component_id, "certificate": certificate, "component": component}


def get_all_existing_config_files():
    """"""
    etc = os.path.join(get_fedbiomed_root(), 'etc', '')
    return [file for file in glob.glob(f"{etc}*.ini")]


def get_all_existing_component_ids():

    config_files = get_all_existing_config_files()
    config_ids = []
    for config in config_files:

        id_, *_ = get_component_config(config)
        config_ids.append(id_)

    return config_ids


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

    for config in config_files:
        component_id, *_ = get_component_from_config(config)
        db_name = f"{DB_PREFIX}{component_id}"
        db_names = {**db_names, component_id: db_name}

    return db_names
