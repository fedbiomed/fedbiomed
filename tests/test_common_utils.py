import configparser
import os

import pytest

from fedbiomed.common.constants import (
    CACHE_FOLDER_NAME,
    CONFIG_FOLDER_NAME,
    TMP_FOLDER_NAME,
    VAR_FOLDER_NAME,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.utils._config_utils import (
    _get_fedbiomed_root,
    create_fedbiomed_setup_folders,
    get_all_existing_certificates,
    get_all_existing_config_files,
    get_component_certificate_from_config,
    get_component_config,
    get_existing_component_db_names,
)


@pytest.fixture()
def write_component(tmp_path, mocker):
    """Writes component config files in a temporary `etc` directory.

    The directory also stands in for `ROOT_DIR`, so the functions that discover
    components in the Fed-BioMed root find the written components instead of the
    ones installed on the machine running the tests.
    """
    mocker.patch("fedbiomed.common.utils._config_utils.ROOT_DIR", str(tmp_path))

    etc = tmp_path / CONFIG_FOLDER_NAME
    etc.mkdir()

    def _write(component_id, component="NODE", certificate="test-certificate"):
        cfg = configparser.ConfigParser()
        cfg["default"] = {"id": component_id, "component": component}
        # Certificate paths are relative to the directory holding the config
        cfg["certificate"] = {
            "public_key": os.path.join("certs", f"{component_id}.pem")
        }

        config_path = etc / f"{component_id}.ini"
        with open(config_path, "w") as file:
            cfg.write(file)

        if certificate is not None:
            cert_path = etc / "certs" / f"{component_id}.pem"
            cert_path.parent.mkdir(exist_ok=True)
            cert_path.write_text(certificate)

        return str(config_path)

    return _write


def test_get_fedbiomed_root(mocker):
    """Root is the directory holding `envs`, searched from the installed package"""
    listdir = mocker.patch("fedbiomed.common.utils._config_utils.os.listdir")

    listdir.return_value = ["envs"]
    root = _get_fedbiomed_root()
    assert os.path.isdir(root)

    # Without `envs` the package is installed, and the root is one level up
    listdir.return_value = []
    assert _get_fedbiomed_root() == os.path.dirname(root)


def test_get_component_config(write_component):
    config = get_component_config(write_component("node-1"))
    assert config["default"]["id"] == "node-1"


def test_get_component_config_raises_for_missing_file(tmp_path):
    with pytest.raises(FedbiomedError):
        get_component_config(str(tmp_path / "missing.ini"))


def test_get_component_config_raises_for_malformed_file(tmp_path):
    config_path = tmp_path / "malformed.ini"
    config_path.write_text("id = node-1")  # no section header

    with pytest.raises(FedbiomedError):
        get_component_config(str(config_path))


def test_get_component_certificate_from_config(write_component):
    config_path = write_component("node-1", certificate="test-certificate")

    assert get_component_certificate_from_config(config_path) == {
        "party_id": "node-1",
        "certificate": "test-certificate",
        "component": "NODE",
    }


def test_get_component_certificate_from_config_upper_cases_component(write_component):
    # Component types are compared against `ComponentType` names, which are
    # uppercase; a config written otherwise reads back the same way
    config_path = write_component("node-1", component="node")

    assert get_component_certificate_from_config(config_path)["component"] == "NODE"


def test_get_component_certificate_from_config_raises_for_missing_certificate(
    write_component,
):
    config_path = write_component("node-1", certificate=None)

    with pytest.raises(FedbiomedError):
        get_component_certificate_from_config(config_path)


def test_get_component_certificate_from_config_raises_for_incomplete_config(tmp_path):
    config_path = tmp_path / "incomplete.ini"
    config_path.write_text("[default]\nid = node-1\n")  # no certificate section

    with pytest.raises(FedbiomedError):
        get_component_certificate_from_config(str(config_path))


def test_get_all_existing_config_files(tmp_path, write_component):
    config_path = write_component("node-1")
    (tmp_path / CONFIG_FOLDER_NAME / "not-a-config.txt").write_text("")

    assert get_all_existing_config_files() == [config_path]


def test_get_all_existing_certificates(write_component):
    write_component("node-1", certificate="test-certificate-1")
    write_component("node-2", certificate="test-certificate-2")

    certificates = sorted(get_all_existing_certificates(), key=lambda c: c["party_id"])

    assert certificates == [
        {
            "party_id": "node-1",
            "certificate": "test-certificate-1",
            "component": "NODE",
        },
        {
            "party_id": "node-2",
            "certificate": "test-certificate-2",
            "component": "NODE",
        },
    ]


def test_get_existing_component_db_names(write_component):
    write_component("node-1")
    write_component("node-2")

    assert get_existing_component_db_names() == {
        "node-1": "db_node-1",
        "node-2": "db_node-2",
    }


def test_create_fedbiomed_setup_folders(tmp_path):
    folders = create_fedbiomed_setup_folders(str(tmp_path))

    var_dir = tmp_path / VAR_FOLDER_NAME
    assert folders == (
        str(tmp_path / CONFIG_FOLDER_NAME),
        str(var_dir),
        str(var_dir / CACHE_FOLDER_NAME),
        str(var_dir / TMP_FOLDER_NAME),
    )
    assert all(os.path.isdir(folder) for folder in folders)

    # Setting up an already existing component leaves the folders in place
    assert create_fedbiomed_setup_folders(str(tmp_path)) == folders
