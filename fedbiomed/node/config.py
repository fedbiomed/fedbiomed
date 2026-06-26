# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, Optional

from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Component, Config
from fedbiomed.common.constants import (
    DEFAULT_CERT_NAME,
    DEFAULT_NODE_ALIAS,
    DEFAULT_NODE_NAME,
    NODE_DATA_FOLDER,
    NODE_DYNAMIC_DATA_FOLDER,
    HashingAlgorithms,
    __node_config_version__,
)
from fedbiomed.common.logger import logger

NODE_CONFIG_SECURITY_SECTION = "security"
NODE_CONFIG_SECURITY_FIELDS = {
    "hashing_algorithm": {
        "type": "enum",
        "default": HashingAlgorithms.SHA256.value,
        "options": [algorithm.value for algorithm in HashingAlgorithms],
    },
    "allow_default_training_plans": {
        "type": "boolean",
        "default": "True",
        "env": "FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS",
    },
    "training_plan_approval": {
        "type": "boolean",
        "default": "False",
        "env": "FBM_SECURITY_TRAINING_PLAN_APPROVAL",
    },
    "secure_aggregation": {
        "type": "boolean",
        "default": "True",
        "env": "FBM_SECURITY_SECURE_AGGREGATION",
    },
    "force_secure_aggregation": {
        "type": "boolean",
        "default": "False",
        "env": "FBM_SECURITY_FORCE_SECURE_AGGREGATION",
    },
    "secagg_insecure_validation": {
        "type": "boolean",
        "default": "True",
        "env": "FBM_SECURITY_SECAGG_INSECURE_VALIDATION",
    },
    "allow_preproc": {
        "type": "boolean",
        "default": "True",
        "env": "FBM_SECURITY_ALLOW_PREPROC",
    },
    "allow_federated_analytics": {
        "type": "boolean",
        "default": "True",
        "env": "FBM_SECURITY_ALLOW_FEDERATED_ANALYTICS",
    },
    "minimum_samples": {
        "type": "integer",
        "default": "0",
        "env": "FBM_SECURITY_MINIMUM_SAMPLES",
        "min": 0,
    },
}
NODE_CONFIG_READ_ONLY_FIELDS = {
    ("default", "id"),
    ("default", "db"),
}
NODE_CONFIG_READ_ONLY_SECTIONS = {
    "certificate",
}
NODE_CONFIG_SKIPPED_SECTIONS = set()
NODE_CONFIG_FIELD_SCHEMAS = {
    NODE_CONFIG_SECURITY_SECTION: NODE_CONFIG_SECURITY_FIELDS,
    "researcher": {
        "port": {
            "type": "integer",
            "min": 0,
        },
    },
    "syslog": {
        "enable": {
            "type": "boolean",
        },
        "port": {
            "type": "integer",
            "min": 0,
        },
    },
}


class NodeConfig(Config):
    _CONFIG_VERSION: str = __node_config_version__
    COMPONENT_TYPE: str = "NODE"

    def __init__(self, *args, alias: Optional[str] = DEFAULT_NODE_ALIAS, **kwargs):
        """NodeConfig constructor

        Args:
            *args: Positional arguments for the parent class `Config`
            alias (str): Alias for the component, used to identify the
                component in the configuration
            **kwargs: Keyword arguments for the parent class `Config`
        """

        self._component_alias = alias
        # Call the parent class constructor after setting the component alias
        super().__init__(*args, **kwargs)

    def add_parameters(self):
        """Generate `Node` config"""

        self._cfg["default"]["name"] = self._component_alias

        # Security variables
        self._cfg[NODE_CONFIG_SECURITY_SECTION] = {
            key: os.getenv(field.get("env", ""), str(field["default"]))
            for key, field in NODE_CONFIG_SECURITY_FIELDS.items()
        }
        # Generate self-signed certificates
        key_file, pem_file = generate_certificate(
            root=self.root,
            component_id=self._cfg["default"]["id"],
            prefix=DEFAULT_CERT_NAME,
        )

        self._cfg["certificate"] = {
            "private_key": os.path.relpath(key_file, os.path.join(self.root, "etc")),
            "public_key": os.path.relpath(pem_file, os.path.join(self.root, "etc")),
        }

        # gRPC server host and port
        self._cfg["researcher"] = {
            "ip": os.getenv("FBM_RESEARCHER_IP", "localhost"),
            "port": os.getenv("FBM_RESEARCHER_PORT", "50051"),
        }

    @staticmethod
    def _infer_gui_field_type(value: str) -> str:
        """Infer a GUI field type for config values without explicit schema.

        Args:
            value: Raw string value read from `config.ini`.

        Returns:
            Best-effort field type for GUI rendering. Boolean-like strings are
            exposed as `boolean`, integer-like strings are exposed as
            `integer`, and all other values are exposed as `string`.
        """

        normalized = value.strip().lower()
        if normalized in {"true", "false", "1", "0", "yes", "no"}:
            return "boolean"

        try:
            int(value)
            return "integer"
        except ValueError:
            return "string"

    def get_gui_config_sections(self) -> Dict[str, Dict[str, Any]]:
        """Return current config sections and field descriptors for the GUI.

        This method is the single owner of GUI-editable node config metadata.
        It derives sections and keys from the loaded `ConfigParser` instance,
        skips sections that should not be edited from the GUI, applies explicit
        schema hints where the node config has known types or constraints, and
        marks immutable fields as read-only.

        Returns:
            Mapping of section names to section descriptors. Each descriptor
            contains a human-readable label and a `fields` mapping. Each field
            descriptor contains the field type, label, editability flag, and
            optional validation metadata such as enum options or integer
            minimum value.
        """

        sections = {}
        for section in self.sections():
            if section in NODE_CONFIG_SKIPPED_SECTIONS:
                continue

            fields = {}
            for key in self._cfg.options(section):
                value = self.get(section, key)
                schema = dict(NODE_CONFIG_FIELD_SCHEMAS.get(section, {}).get(key, {}))
                schema.setdefault("type", self._infer_gui_field_type(value))
                schema.setdefault("label", key.replace("_", " ").title())
                schema["editable"] = (
                    section not in NODE_CONFIG_READ_ONLY_SECTIONS
                    and (section != "default" or key == "name")
                    and (section, key) not in NODE_CONFIG_READ_ONLY_FIELDS
                )
                fields[key] = schema

            sections[section] = {
                "label": section.replace("_", " ").title(),
                "fields": fields,
            }

        return sections

    def migrate(self):
        """Please add migrated parameters for the new version.

        See [`Config.migrate`][fedbiomed.common.config.Config.migrate] for more information
        """
        if not self._cfg.has_option("default", "name"):
            logger.warning(
                "DEPRECATION: You are using an old configuration file for the node. "
                "Please add 'name' value in `default` section "
                "of the node configuration to define a name."
            )

            self._cfg["default"].update({"name": "Migrated Node Name"})

        if not self._cfg.has_option("security", "allow_preproc"):
            logger.warning(
                "DEPRECATION: You are using an old configuration file for the node. "
                "Please add 'allow_preproc' value in `security` section "
                "of the node configuration to define whether preprocessing is allowed."
            )

            self._cfg["security"].update({"allow_preproc": "True"})

        if not self._cfg.has_option("security", "allow_federated_analytics"):
            logger.warning(
                "DEPRECATION: You are using an old configuration file for the node. "
                "Please add 'allow_federated_analytics' value in `security` section "
                "of the node configuration to define whether federated analytics is allowed."
            )

            self._cfg["security"].update({"allow_federated_analytics": "True"})

        if not self._cfg.has_option("security", "minimum_samples"):
            logger.warning(
                "DEPRECATION: You are using an old configuration file for the node. "
                "Please add 'minimum_samples' value in `security` section "
                "of the node configuration to define the minimum number of samples required for a dataset."
            )
            self._cfg["security"].update({"minimum_samples": "0"})

        if not self._cfg.has_section("syslog"):
            logger.warning(
                "DEPRECATION: You are using an old configuration file for the node. "
                "Please add 'enable' value in `syslog` section "
                "of the node configuration to define whether syslog is enabled."
            )
            self._cfg.add_section("syslog")
            self._cfg["syslog"].update({"enable": "False"})
            self._cfg["syslog"].update({"host": "localhost"})
            self._cfg["syslog"].update({"port": "514"})
            self._cfg["syslog"].update({"protocol": "udp"})
            self._cfg["syslog"].update({"facility": "user"})
            self._cfg["syslog"].update({"level": "INFO"})


component_root = os.environ.get("FBM_NODE_COMPONENT_ROOT", None)


class NodeComponent(Component):
    """Fed-BioMed Node Component Class

    This class is used for creating and validating components
    by given component root directory
    """

    config_cls = NodeConfig
    _default_component_name = DEFAULT_NODE_NAME

    def initiate(
        self,
        root: Optional[str] = None,
        alias: Optional[str] = DEFAULT_NODE_ALIAS,
    ) -> NodeConfig:
        """Initiates the Node component

        Args:
            root (str, optional): Root directory for the component. If None, uses the default.
            alias (str, optional): Alias for the component, used to identify the component in the configuration.

        Returns:
            NodeConfig: The configuration object for the Node component.
        """
        config = super().initiate(root=root, alias=alias)
        config.write()
        node_data_path = os.path.join(config.root, NODE_DATA_FOLDER)
        os.makedirs(node_data_path, exist_ok=True)
        node_dynamic_data_path = os.path.join(config.root, NODE_DYNAMIC_DATA_FOLDER)
        os.makedirs(node_dynamic_data_path, exist_ok=True)
        return config


node_component = NodeComponent()
