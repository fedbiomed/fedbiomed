# Security Logging in FedLogger

## Overview

Security logs are audit-grade records of security-relevant activity. 

They serve a different purpose than application logs:

Application logs help developers debug and operate the system (errors, warnings, performance, flow).
Security logs provide an audit trail of sensitive actions and security events (who did what, when, and with what outcome).

Security logs should be:

- Structured: machine-readable JSON for analysis, alerting, and compliance.
- Complete: include enough context to reconstruct events.
- Tamper-resistant: written to a dedicated file with rotation, separated from normal logs.
- Predictable: stable schema so downstream tooling can parse reliably.

In Fedbiomed, security logs are written through the same logger API, but are routed to a dedicated handler (`SECURITY_FILE`) and formatted by a dedicated formatter (`_SecurityFormatter`).

## What counts as a “security event”?

A security event is any action that should be auditable, such as: 

- All messages coming and going to/from the node. 
    - Messages to/from researcher(s).
    - Node2node communication (For secagg, diffie-hellman key exchanges...)
- Creation, deletion, start, stop of a node.
- CRUD operations on all the datasets.
- CRUD operations and approvals/rejections for training plans.
- Creation, deletion, start, stop, save of an experiment.
- Information regarding training/round. Info on each training step, dataset split...
- CRUD operations on certificates.
- CRUD operations on config files.
- Optional: Info regarding secagg operations.

## Security logging architecture

### Dedicated handler and filter

Security logs are written to a dedicated handler registered under the type key `SECURITY_FILE`.

This handler is configured with:

- A file destination
- Rotation policy (e.g., timed rotation)
- A filter that ensures only security events are written.

This filter is conceptually:

If `record.is_security == True` → allow it into the security log.

Otherwise → reject it.

This ensures normal application logging never contaminates the security audit log.

### Dedicated JSON formatter and Default Fields

Security logs use a dedicated JSON formatter (`_SecurityFormatter`) that produces a stable schema including event fields like:

- timestamp
- node metadata / defaults (node id/name, version)
- operation
- status
- experiment_id, researcher_id, round_number etc.

## How to write security logs

There are two supported patterns:

1) Use `security_event()` for canonical security events (recommended)

Use `security_event()` when you’re recording a structured security/audit event. This path is the most stable and produces consistent JSON.

Example:

```python
logger.security_event(
    operation="training_plan_approved",
    status="success",
    researcher_id="researcher_1",
    round_number=3,
    dataset_id="ds_42",
    custom_field="custom_value",
)
```

This automatically creates a structured JSON security entry with a predictable schema. It handles the routing of the entry to the security file handler and ensures it is marked as a security record (`is_security=True`) so it passes the security filter.

2) Mark an application log record as security-relevant with `extra={"is_security": True}`

Sometimes you want to log a message and have it appear in both normal logs and the security audit log, or you want to record a human-readable message as a security record.

You can do that by passing the is_security flag in extra:

```python
logger.info(
    "Researcher joined the federation",
    extra={
        "is_security": True,
        "operation": "researcher_join",
        "status": "success",
        "researcher_id": "researcher_7",
    },
)
```

!!! warning "Important" 
    `Operation` is a required field in security logs. It is used as a stable identifier to separate security logs from application logs. Any security log without an `operation` field will raise an error.

## Setting shared fields with Security Context

Many security events share the same context fields (e.g., researcher_id, round_number, component, session_id). Instead of passing them repeatedly, you can set a common context once using `security_context()` and then emit multiple events.

Example
```python
# Set common context once (applies to subsequent security events)
logger.security_context(
    researcher_id="researcher_1",
    round_number=12,
    component="node",
)

# Emit multiple security events without repeating the same fields
logger.security_event(
    operation="training_start",
    status="initiated",
)

logger.security_event(
    operation="training_end",
    status="success",
    duration_ms=8342,
)
```

## Configuring the security log file

### Default location

If you do not provide a filename when creating the security file handler, security logs are written to `log/security_audit.log` in the node folder. This gives you a predictable default audit location without additional configuration.

If you want to set a custom path:

```python
logger.add_security_file_handler(filename="/var/log/fedbiomed/security_audit.log")
```

After adding the handler, ensure it is enabled at the intended level. If your logger implementation tracks a “default” handler level separately, explicitly set the handler level:

```python
logger.setLevel("INFO", "SECURITY_FILE")
```

This avoids cases where security events are emitted at INFO but dropped because the handler is configured at WARNING.

### Rotation (Log Rollover)

The security file handler supports daily timed rotation by default. Rollover to a new file each day avoids log files growing too large and ensures.

## Syslog integration

### What is Syslog?

Syslog is a standard protocol used to send log records to a central log collector or SIEM. Instead of keeping audit information only in the local `security_audit.log` file, a node can also forward security events to a remote syslog server for aggregation, retention, monitoring, and alerting.

In Fedbiomed, syslog is an optional output for security logs. When enabled, FedLogger adds a dedicated `SYSLOG` handler that sends security events to the configured remote syslog endpoint.

### What it does in Fedbiomed

When syslog is enabled:

- Only security events are forwarded to syslog.
- The same security filter is applied as for the dedicated security log file.
- The same structured JSON security formatter is used, so the payload stays aligned with the local security audit log.
- Events are sent to the configured remote host and port over UDP or TCP.

When syslog is disabled, no remote syslog handler is created and security events remain local unless another handler is configured.

### How to configure syslog in `config.ini`

Syslog is configured through the `[syslog]` section of the component `config.ini` file.

Example:

```ini
[syslog]
enable = True
host = localhost
port = 514
protocol = udp
facility = user
level = INFO
```

Available fields:

- `enable`: enables or disables syslog forwarding. Accepted values follow the usual boolean config parsing (`True`, `False`, `1`, etc.).
- `host`: hostname or IP address of the remote syslog server.
- `port`: remote syslog port.
- `protocol`: transport protocol. Supported values are `udp` and `tcp`.
- `facility`: syslog facility name. This must match one of the facilities supported by Python's `SysLogHandler`, such as `user`, `daemon`, `local0`, ..., `local7`.
- `level`: minimum Fedbiomed log level that the syslog handler will emit. This must match the logging levels supported by the Python library, which are `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`.

If `protocol`, `facility`, or `level` contains an unsupported value, configuration loading raises an error.

### Defaults and behavior when not configured

If the `[syslog]` section is missing or `enable = False`, syslog forwarding is disabled by default.

If syslog is enabled but some values are omitted in the `config.ini` file, Fedbiomed uses these defaults:

- `host = localhost`
- `port = 514`
- `protocol = udp`
- `facility = user`
- `level = INFO`

These are the default values used for forwarding the logs to the local system level syslog process, which writes the logs to the system level default file. This file location can change depending on the UNIX distribution, it is usually in one of:

- `var/log/syslog`
- `var/log/messages`

For node configuration files, recent configuration migration code also writes these same defaults into the `[syslog]` section in the `config.ini` file when upgrading the older configuration files.

## Practical guidance and conventions

### 1) Required event fields

For meaningful security logs, include:

- operation (required): stable event name (researcher_join, dataset_access, training_start, training_stop, auth_failed, etc.)
- status: success, failure, denied, etc.
- researcher_id: if action is tied to a researcher
- additional identifiers as needed (dataset id, round_number, etc.)

Information regarding the node (node id, node name), fedbiomed version and timestamp are included by default.

### 2) Don’t log secrets

!!! warning "Never include:"
    - Credentials, tokens, keys, passwords
    - Personal data unless necessary and approved
    - Full exception payloads of sensitive objects

## Summary

- Use security logs for auditability, not debugging.
- Operation is required for every security log entry.
- Prefer logger.security_event(...) for canonical, structured events.
- Use logger.security_context(...) to set shared fields for a sequence of events.
- Use extra={"is_security": True, ...} for “also-security” log entries when needed.
- If no filename is provided, security logs default to log/security_audit.log in the node folder.