import configparser
from unittest.mock import patch

import pytest
from packaging.version import Version

from fedbiomed.common.config import Config
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedConfigurationError,
)
from fedbiomed.common.logger import SYSLOG_FACILITY_MAP
from fedbiomed.node.config import NodeComponent, NodeConfig
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


@pytest.fixture()
def config_env(tmp_path):
    """Gives a fresh config root, patches filesystem side effects of config
    generation, and restores the class-level `Config.vars` shared with any
    already existing config instance (e.g. the researcher config singleton)."""
    saved_vars = dict(Config.vars)
    with (
        patch("fedbiomed.common.config.create_fedbiomed_setup_folders"),
        patch("builtins.open"),
    ):
        yield tmp_path
    Config.vars.clear()
    Config.vars.update(saved_vars)


@pytest.fixture()
def concrete_config(config_env):
    """Makes the abstract Config instantiable; yields the config root."""
    with patch.multiple("fedbiomed.common.config.Config", __abstractmethods__=set()):
        yield config_env


def test_config_read_rejects_an_incompatible_version(concrete_config):
    """A configuration written by another major version is refused."""
    config = Config(root=str(concrete_config))
    config._cfg["default"] = {"version": "1.0"}
    config._CONFIG_VERSION = Version("99.0")

    with patch("fedbiomed.common.config.logger.security_event") as security_event:
        with pytest.raises(FedbiomedConfigurationError):
            config.read()

    # The raise is generic; the version that was refused is in the audit event
    (failure,) = [
        c
        for c in security_event.call_args_list
        if c.kwargs.get("operation") == "config_read"
        and c.kwargs.get("status") == "failure"
    ]
    assert "found version 1.0 expected version 99.0" in failure.kwargs["error_message"]


def test_config_generates_when_none_exists(concrete_config):
    with patch("fedbiomed.common.config.Config.generate") as generate:
        Config(root=str(concrete_config))

    generate.assert_called_once()


def test_config_is_config_existing(concrete_config):
    config = Config(root=str(concrete_config))
    assert not config.is_config_existing()


def test_node_config_generate(config_env):
    config = NodeConfig(root=str(config_env))

    component = config.get("default", "component")
    assert component.lower() == "node"

    assert config.get("researcher", "ip")
    assert config.get("researcher", "port")


def test_node_config_sections(config_env):
    config = NodeConfig(root=str(config_env))

    sections = config.sections()

    assert "researcher" in sections
    assert "default" in sections
    assert "security" in sections
    assert "certificate" in sections
    assert "authentication" in sections


def test_node_config_authentication_section_defaults(config_env):
    config = NodeConfig(root=str(config_env))

    # Opt-in: disabled by default. Trusted certs live in the main component DB.
    assert not config.getbool("authentication", "mutual_authentication")


def test_node_config_migrate_old(config_env):
    config = NodeConfig(root=str(config_env))

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

    assert config._cfg.has_option("default", "name")
    assert config._cfg.has_option("security", "allow_preproc")
    assert config._cfg.has_option("security", "allow_federated_analytics")
    assert config._cfg.has_option("security", "minimum_samples")
    assert config._cfg.get("security", "minimum_samples") == "0"


def test_node_config_migrate_new(config_env):
    config = NodeConfig(root=str(config_env))

    # Should not raise any warning
    with patch("fedbiomed.common.logger.logger.warning") as log_warn:
        config.migrate()
        log_warn.assert_not_called()


@pytest.mark.parametrize("approval", ["True", "False"])
def test_node_config_retires_automatic_approval_and_hash_options(
    tmp_path, approval, monkeypatch
):
    """Old flags disappear from persisted config and GUI without changing approval mode."""
    from fedbiomed_node_api.helpers.config_schema import get_config_sections_schema

    (tmp_path / "etc").mkdir()
    monkeypatch.setattr(Config, "vars", dict(Config.vars))
    monkeypatch.setenv("FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS", "True")
    config = NodeConfig(root=str(tmp_path))
    config.set("default", "version", "2.4.0")
    config.set("security", "training_plan_approval", approval)
    config.set("security", "allow_default_training_plans", "True")
    config.set("security", "hashing_algorithm", "SHA512")
    config.write()

    migrated = NodeConfig(root=str(tmp_path))
    assert migrated.getbool("security", "training_plan_approval") == (
        approval == "True"
    )
    assert migrated.get("default", "version") == "2.5.0"
    for key in ("allow_default_training_plans", "hashing_algorithm"):
        assert not migrated._cfg.has_option("security", key)
        assert key not in get_config_sections_schema(migrated)["security"]["fields"]
    migrated.write()
    reloaded = NodeConfig(root=str(tmp_path))
    assert reloaded.getbool("security", "training_plan_approval") == (
        approval == "True"
    )
    assert not reloaded._cfg.has_option("security", "hashing_algorithm")


def test_new_node_has_no_automatically_approved_plan_options(config_env):
    config = NodeConfig(root=str(config_env))
    assert not config.getbool("security", "training_plan_approval")
    assert not config._cfg.has_option("security", "hashing_algorithm")
    assert not config._cfg.has_option("security", "allow_default_training_plans")


@pytest.mark.parametrize("approval", ["True", "False"])
def test_node_component_persists_legacy_config_migration(
    tmp_path, approval, monkeypatch
):
    """The normal component entry point must not reread away its migration."""
    monkeypatch.setattr(Config, "vars", dict(Config.vars))
    root = tmp_path / "node"
    original = NodeComponent().initiate(str(root))
    original.set("default", "version", "2.4.0")
    original.set("security", "training_plan_approval", approval)
    original.set("security", "hashing_algorithm", "SHA512")
    original.set("security", "allow_default_training_plans", "True")
    original.write()

    for _ in range(2):
        upgraded = NodeComponent().initiate(str(root))
        assert upgraded.get("default", "version") == "2.5.0"
        assert upgraded.getbool("security", "training_plan_approval") == (
            approval == "True"
        )
        persisted = configparser.ConfigParser()
        persisted.read(upgraded.config_path)
        assert persisted["default"]["version"] == "2.5.0"
        assert persisted.getboolean("security", "training_plan_approval") == (
            approval == "True"
        )
        for key in ("hashing_algorithm", "allow_default_training_plans"):
            assert not upgraded._cfg.has_option("security", key)
            assert not persisted.has_option("security", key)


def test_researcher_config_generate(config_env):
    config = ResearcherConfig(root=str(config_env))

    component = config.get("default", "component")
    assert component.lower() == "researcher"

    assert config.get("server", "host")
    assert config.get("server", "port")
    assert config.get("certificate", "private_key")
    assert config.get("certificate", "public_key")


def test_researcher_config_sections(config_env):
    config = ResearcherConfig(root=str(config_env))
    sections = config.sections()

    assert "server" in sections
    assert "default" in sections
    assert "certificate" in sections
    assert "authentication" in sections


def test_researcher_config_authentication_section_defaults(config_env):
    # The researcher is the side that demands client certificates, and does not
    # until it is told to
    config = ResearcherConfig(root=str(config_env))

    assert not config.getbool("authentication", "mutual_authentication")
