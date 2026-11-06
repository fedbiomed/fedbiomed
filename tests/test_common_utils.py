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
    get_existing_component_db_paths,
)


@pytest.fixture()
def write_component(tmp_path):
    """Writes component roots under a temporary directory.

    Each is laid out the way a created component is, so the functions that
    discover components under a directory find the written ones.
    """

    def _write(component_id, component="NODE", certificate="test-certificate"):
        etc = tmp_path / component_id / CONFIG_FOLDER_NAME
        etc.mkdir(parents=True)

        cfg = configparser.ConfigParser()
        # Certificate and database paths are relative to the directory holding
        # the config
        cfg["default"] = {
            "id": component_id,
            "component": component,
            "db": os.path.join("..", VAR_FOLDER_NAME, f"db_{component_id}.json"),
        }
        cfg["certificate"] = {
            "public_key": os.path.join("certs", f"{component_id}.pem")
        }

        config_path = etc / "config.ini"
        with open(config_path, "w") as file:
            cfg.write(file)

        if certificate is not None:
            cert_path = etc / "certs" / f"{component_id}.pem"
            cert_path.parent.mkdir()
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
        "component_id": "node-1",
        "certificate": "test-certificate",
        "component_type": "NODE",
    }


def test_get_component_certificate_from_config_upper_cases_component(write_component):
    # Component types are compared against `ComponentType` names, which are
    # uppercase; a config written otherwise reads back the same way
    config_path = write_component("node-1", component="node")

    assert (
        get_component_certificate_from_config(config_path)["component_type"] == "NODE"
    )


def test_get_component_certificate_from_config_raises_for_missing_certificate(
    write_component,
):
    config_path = write_component("node-1", certificate=None)

    with pytest.raises(FedbiomedError):
        get_component_certificate_from_config(config_path)


@pytest.mark.parametrize(
    "content",
    [
        "[default]\nid = node-1\n",  # no `component`
        "[default]\nid = node-1\ncomponent = NODE\n",  # no `[certificate]` section
    ],
)
def test_get_component_certificate_from_config_raises_for_incomplete_config(
    tmp_path, content
):
    """Every declaration the certificate is read from, missing one at a time."""
    config_path = tmp_path / "incomplete.ini"
    config_path.write_text(content)

    with pytest.raises(FedbiomedError):
        get_component_certificate_from_config(str(config_path))


def test_get_all_existing_config_files(tmp_path, write_component):
    config_path = write_component("node-1")
    # Neither a directory without a component in it nor a file is a component
    (tmp_path / "not-a-component").mkdir()
    (tmp_path / "not-a-component.txt").write_text("")

    assert get_all_existing_config_files(str(tmp_path)) == [config_path]


def test_get_all_existing_config_files_ignores_components_deeper_than_first_level(
    tmp_path,
):
    nested = tmp_path / "parent" / "node-1" / CONFIG_FOLDER_NAME
    nested.mkdir(parents=True)
    (nested / "config.ini").write_text("[default]\nid = node-1\n")

    assert get_all_existing_config_files(str(tmp_path)) == []


def test_get_all_existing_certificates(tmp_path, write_component):
    write_component("node-1", certificate="test-certificate-1")
    write_component("node-2", certificate="test-certificate-2")

    certificates = sorted(
        get_all_existing_certificates(str(tmp_path)), key=lambda c: c["component_id"]
    )

    assert certificates == [
        {
            "component_id": "node-1",
            "certificate": "test-certificate-1",
            "component_type": "NODE",
        },
        {
            "component_id": "node-2",
            "certificate": "test-certificate-2",
            "component_type": "NODE",
        },
    ]


def test_get_existing_component_db_paths(tmp_path, write_component):
    write_component("node-1")
    write_component("node-2")

    assert get_existing_component_db_paths(str(tmp_path)) == {
        "node-1": str(tmp_path / "node-1" / VAR_FOLDER_NAME / "db_node-1.json"),
        "node-2": str(tmp_path / "node-2" / VAR_FOLDER_NAME / "db_node-2.json"),
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
