import configparser
import shutil
import unittest
from unittest.mock import patch

import pytest
from packaging.version import Version

from fedbiomed.common.config import Config
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedConfigurationError,
)
from fedbiomed.common.logger import SYSLOG_FACILITY_MAP
from fedbiomed.node.config import NodeConfig
from fedbiomed.researcher.config import ResearcherConfig

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def patch_security_logger(mocker):
    """Patch security logging hooks in the module where Config uses them."""
    set_security_logs = mocker.patch("fedbiomed.common.config.logger.set_security_logs")
    security_event = mocker.patch("fedbiomed.common.config.logger.security_event")
    mocker.patch("fedbiomed.common.config.logger.setLevel")
    return set_security_logs, security_event


@pytest.fixture()
def DummyConfig():
    """Concrete Config for tests (avoid patching abstractmethods)."""

    class _DummyConfig(Config):
        _CONFIG_VERSION = Version("1.0")
        COMPONENT_TYPE = "DUMMY"

        def add_parameters(self):
            self._cfg["dummy"] = {"x": "y"}

        def migrate(self):
            return

    return _DummyConfig


@pytest.fixture()
def patch_cert_generation(mocker, tmp_path):
    """Avoid crypto/fs side effects for Node/Researcher configs."""
    mocker.patch(
        "fedbiomed.node.config.generate_certificate",
        lambda **kwargs: (str(tmp_path / "k.key"), str(tmp_path / "c.pem")),
    )
    mocker.patch(
        "fedbiomed.researcher.config.generate_certificate",
        lambda **kwargs: (str(tmp_path / "k.key"), str(tmp_path / "c.pem")),
    )


###################################
##### Tests for security logging
###################################


def test_security_logging_called_on_init_load_generate(
    tmp_path, mocker, patch_security_logger, DummyConfig
):
    set_security_logs, security_event = patch_security_logger

    mocker.patch.object(DummyConfig, "is_config_existing", return_value=False)
    DummyConfig(root=str(tmp_path))

    # logger.set_security_logs(root_path=...) called
    set_security_logs.assert_called()
    assert set_security_logs.call_args.kwargs["root_path"] == str(tmp_path)

    # config_load_start initiated
    assert any(
        c.kwargs.get("operation") == "config_load_start"
        and c.kwargs.get("status") == "initiated"
        for c in security_event.call_args_list
    )

    # config_generate success
    assert any(
        c.kwargs.get("operation") == "config_generate"
        and c.kwargs.get("status") == "success"
        for c in security_event.call_args_list
    )


def test_security_logging_config_set_emits_event(
    tmp_path, mocker, patch_security_logger, DummyConfig
):
    _, security_event = patch_security_logger
    mocker.patch.object(DummyConfig, "is_config_existing", return_value=False)

    cfg = DummyConfig(root=str(tmp_path))
    cfg.set("dummy", "x", "z")

    assert any(
        c.kwargs.get("operation") == "config_set"
        and c.kwargs.get("status") == "success"
        and c.kwargs.get("config_section") == "dummy"
        and c.kwargs.get("config_key") == "x"
        for c in security_event.call_args_list
    )


def test_security_logging_write_success_and_failure(
    tmp_path, mocker, patch_security_logger, DummyConfig
):
    _, security_event = patch_security_logger
    mocker.patch.object(DummyConfig, "is_config_existing", return_value=False)

    mocker.patch("fedbiomed.common.config.create_fedbiomed_setup_folders")
    mocker.patch("fedbiomed.common.config.open", mocker.mock_open())

    cfg = DummyConfig(root=str(tmp_path))

    # Success
    cfg.write()
    assert any(
        c.kwargs.get("operation") == "config_write"
        and c.kwargs.get("status") == "success"
        for c in security_event.call_args_list
    )

    # Failure: ConfigParser.write raises configparser.Error
    def boom(*args, **kwargs):
        raise configparser.Error("boom")

    mocker.patch.object(cfg._cfg, "write", side_effect=boom)

    with pytest.raises(FedbiomedConfigurationError):
        cfg.write()

    assert any(
        c.kwargs.get("operation") == "config_write"
        and c.kwargs.get("status") == "failure"
        and "error_message" in c.kwargs
        for c in security_event.call_args_list
    )


def test_add_syslog_from_config_registers_handler(tmp_path, mocker, DummyConfig):
    mocker.patch.object(DummyConfig, "is_config_existing", return_value=False)
    add_syslog_handler = mocker.patch(
        "fedbiomed.common.config.logger.add_syslog_handler"
    )

    cfg = DummyConfig(root=str(tmp_path))
    cfg._cfg["syslog"] = {
        "enable": "True",
        "host": "syslog.example",
        "port": "1514",
        "protocol": "tcp",
        "facility": "local3",
        "level": "warning",
    }

    cfg.add_syslog_from_config()

    add_syslog_handler.assert_called_once_with(
        host="syslog.example",
        port=1514,
        protocol="tcp",
        facility=SYSLOG_FACILITY_MAP["local3"],
        level="WARNING",
    )


@pytest.mark.parametrize(
    "option,value,error_suffix",
    [
        ("protocol", "http", "Unsupported syslog protocol: http"),
        ("facility", "badfacility", "Unsupported syslog facility: badfacility"),
        ("level", "TRACE", "Unsupported syslog level: TRACE"),
    ],
)
def test_add_syslog_from_config_rejects_invalid_values(
    tmp_path, mocker, DummyConfig, option, value, error_suffix
):
    mocker.patch.object(DummyConfig, "is_config_existing", return_value=False)
    mocker.patch("fedbiomed.common.config.logger.add_syslog_handler")

    cfg = DummyConfig(root=str(tmp_path))
    cfg._cfg["syslog"] = {
        "enable": "True",
        "host": "localhost",
        "port": "514",
        "protocol": "udp",
        "facility": "user",
        "level": "INFO",
    }
    cfg._cfg["syslog"][option] = value

    with pytest.raises(FedbiomedConfigurationError) as exc_info:
        cfg.add_syslog_from_config()

    assert str(exc_info.value) == f"{ErrorNumbers.FB600.value}: {error_suffix}"


class BaseConfigTest(unittest.TestCase):
    def setUp(self):
        self.patch_fed_folders = patch(
            "fedbiomed.common.config.create_fedbiomed_setup_folders"
        )
        self.patch_open = patch("builtins.open")

        self.open_mock = self.patch_open.start()
        self.create_fed_folders_mock = self.patch_fed_folders.start()

    def tearDown(self):
        self.patch_open.stop()
        self.patch_fed_folders.stop()

        # Clean up any created folders
        shutil.rmtree("dummy-root", ignore_errors=True)
        shutil.rmtree("etc", ignore_errors=True)
        return super().tearDown()


class TestConfig(BaseConfigTest):
    """Tests base methods of Config class"""

    def setUp(self):
        self.patch_ = patch.multiple(
            "fedbiomed.common.config.Config", __abstractmethods__=set()
        )
        self.patch_.start()

        return super().setUp()

    def tearDown(self):
        self.patch_.stop()
        return super().tearDown()

    @patch("fedbiomed.common.config.configparser.ConfigParser")
    def test_01_read(self, config_parser):
        config = Config(root="dummy-root")
        config._CONFIG_VERSION = "0.99"

        with self.assertRaises(FedbiomedConfigurationError):
            config.read()

        config_parser.return_value.read.assert_called_once()

        # With autogenereate
        with patch("fedbiomed.common.config.Config.generate") as gen:
            config = Config(root="dummy-root")
            gen.assert_called_once()

    def test_03_is_config_existing(self):
        config = Config(root="test")
        r = config.is_config_existing()
        self.assertFalse(r)


class TestNodeConfig(BaseConfigTest):
    def test_01_node_config_generate(self):
        config = NodeConfig(root="tests")
        print(config._cfg["default"])

        component = config.get("default", "component")
        self.assertEqual("node", component.lower())

        r_ip = config.get("researcher", "ip")
        r_port = config.get("researcher", "port")

        self.assertTrue(r_ip)
        self.assertTrue(r_port)

    def test_02_node_config_sections(self):
        config = NodeConfig(root="test")

        sections = config.sections()

        self.assertTrue("researcher" in sections)
        self.assertTrue("default" in sections)
        self.assertTrue("security" in sections)
        self.assertTrue("certificate" in sections)

    def test_03_node_config_migrate_old(self):
        config = NodeConfig(root="test")

        # Simulate old config by removing options
        if config._cfg.has_option("default", "name"):
            config._cfg.remove_option("default", "name")
        if config._cfg.has_option("security", "allow_preproc"):
            config._cfg.remove_option("security", "allow_preproc")
        if config._cfg.has_option("security", "allow_federated_analytics"):
            config._cfg.remove_option("security", "allow_federated_analytics")
        if config._cfg.has_option("security", "minimum_samples"):
            config._cfg.remove_option("security", "minimum_samples")

        config.migrate()

        self.assertTrue(config._cfg.has_option("default", "name"))
        self.assertTrue(config._cfg.has_option("security", "allow_preproc"))
        self.assertTrue(config._cfg.has_option("security", "allow_federated_analytics"))
        self.assertTrue(config._cfg.has_option("security", "minimum_samples"))
        self.assertEqual(config._cfg.get("security", "minimum_samples"), "0")

    def test_04_node_config_migrate_new(self):
        config = NodeConfig(root="test")

        # Should not raise any warning
        with patch("fedbiomed.common.logger.logger.warning") as log_warn:
            config.migrate()
            log_warn.assert_not_called()


class TestResearcherConfig(BaseConfigTest):
    def test_01_researcher_config_generate(self):
        config = ResearcherConfig(root="tests")

        component = config.get("default", "component")
        self.assertEqual("researcher", component.lower())

        r_ip = config.get("server", "host")
        r_port = config.get("server", "port")
        r_pem = config.get("certificate", "private_key")
        r_key = config.get("certificate", "public_key")

        self.assertTrue(r_ip)
        self.assertTrue(r_port)
        self.assertTrue(r_pem)
        self.assertTrue(r_key)

    def test_02_researcher_config_sections(self):
        config = ResearcherConfig(root="test")
        sections = config.sections()

        self.assertTrue("server" in sections)
        self.assertTrue("default" in sections)


if __name__ == "__main__":
    unittest.main()
