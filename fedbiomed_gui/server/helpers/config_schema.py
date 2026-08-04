# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from fedbiomed.common.constants import HashingAlgorithms

NODE_CONFIG_SECURITY_SECTION = "security"
NODE_CONFIG_READ_ONLY_FIELDS = {
    ("default", "id"),
    ("default", "db"),
}
NODE_CONFIG_READ_ONLY_SECTIONS = {
    "certificate",
}
NODE_CONFIG_SKIPPED_SECTIONS = set()
NODE_CONFIG_FIELD_SCHEMAS = {
    NODE_CONFIG_SECURITY_SECTION: {
        "hashing_algorithm": {
            "type": "enum",
            "options": [algorithm.value for algorithm in HashingAlgorithms],
        },
        "allow_default_training_plans": {
            "type": "boolean",
        },
        "training_plan_approval": {
            "type": "boolean",
        },
        "secure_aggregation": {
            "type": "boolean",
        },
        "force_secure_aggregation": {
            "type": "boolean",
        },
        "secagg_insecure_validation": {
            "type": "boolean",
        },
        "allow_preproc": {
            "type": "boolean",
        },
        "allow_federated_analytics": {
            "type": "boolean",
        },
        "minimum_samples": {
            "type": "integer",
            "min": 0,
        },
    },
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


def infer_config_field_type(value: str) -> str:
    """Infer a field type for config values without explicit schema.

    Args:
        value: Raw string value read from `config.ini`.

    Returns:
        Best-effort field type for rendering and validation. Boolean-like
        strings are exposed as `boolean`, integer-like strings are exposed as
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


def get_config_sections_schema(node_config: Any) -> Dict[str, Dict[str, Any]]:
    """Return current config sections and field descriptors.

    This helper owns the form/editability metadata used by the GUI server. It
    derives sections and keys from the loaded node config object, applies
    explicit schema hints where known, and marks immutable fields as read-only.

    Args:
        node_config: Loaded node configuration object.

    Returns:
        Mapping of section names to section descriptors. Each descriptor
        contains a human-readable label and a `fields` mapping. Each field
        descriptor contains the field type, label, editability flag, and
        optional validation metadata such as enum options or integer minimum
        value.
    """

    sections = {}
    for section in node_config.sections():
        if section in NODE_CONFIG_SKIPPED_SECTIONS:
            continue

        fields = {}
        for key in node_config._cfg.options(section):
            value = node_config.get(section, key)
            schema = dict(NODE_CONFIG_FIELD_SCHEMAS.get(section, {}).get(key, {}))
            schema.setdefault("type", infer_config_field_type(value))
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
