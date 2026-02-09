# Security Logging in FedLogger

## Overview

Security logs are audit-grade records of security-relevant activity. They exist for a different purpose than application logs:

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

***Important***: `Operation` is a required field in security logs. It is used as a stable identifier to separate security logs from application logs. Any security log without an `operation` field will raise an error.

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

## Practical guidance and conventions

### Required event fields

For meaningful security logs, include:

- operation (required): stable event name (researcher_join, dataset_access, training_start, training_stop, auth_failed, etc.)
- status: success, failure, denied, etc.
- researcher_id: if action is tied to a researcher
- additional identifiers as needed (dataset id, round_number, etc.)

Information regarding the node (node id, node name), fedbiomed version and timestamp are included by default.

### Don’t log secrets

Never include:

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