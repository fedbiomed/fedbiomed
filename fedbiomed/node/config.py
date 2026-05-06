# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.config import Component, Config
from fedbiomed.common.constants import (
    DEFAULT_CERT_NAME,
    DEFAULT_NODE_ALIAS,
    DEFAULT_NODE_NAME,
    NODE_DATA_FOLDER,
    HashingAlgorithms,
    __node_config_version__,
)
from fedbiomed.common.logger import logger


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
        self._cfg["security"] = {
            "hashing_algorithm": HashingAlgorithms.SHA256.value,
            "allow_default_training_plans": os.getenv(
                "FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS", "True"
            ),
            "training_plan_approval": os.getenv(
                "FBM_SECURITY_TRAINING_PLAN_APPROVAL", "False"
            ),
            "secure_aggregation": os.getenv("FBM_SECURITY_SECURE_AGGREGATION", "True"),
            "force_secure_aggregation": os.getenv(
                "FBM_SECURITY_FORCE_SECURE_AGGREGATION", "False"
            ),
            "secagg_insecure_validation": os.getenv(
                "FBM_SECURITY_SECAGG_INSECURE_VALIDATION", "True"
            ),
            "allow_preproc": os.getenv("FBM_SECURITY_ALLOW_PREPROC", "True"),
            "allow_federated_analytics": os.getenv(
                "FBM_SECURITY_ALLOW_FEDERATED_ANALYTICS", "True"
            ),
            "minimum_samples": os.getenv("FBM_SECURITY_MINIMUM_SAMPLES", "0"),
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
        return config


node_component = NodeComponent()
