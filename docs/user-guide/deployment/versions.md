# Versions

Fed-BioMed stores and checks version numbers for several of its components. 
The [semantics of the versions](https://semver.org/) are as follows:

- different **major** version: incompatibility that results in halting the execution immediately
- different **minor** or **micro** version: backward compatibility, provide a warning message if versions are different

This page tracks the version changes for each component, to provide further information when incompatibilities are
detected.

## Configuration files

- researcher configuration file
- node configuration file
- node state
- researcher breakpoint
- messaging protocol

### Node configuration 2.5.0

Training-plan hashes now always use SHA256. The `hashing_algorithm` and
`allow_default_training_plans` options are removed when loading an older node
configuration. The configured `training_plan_approval` value is preserved.
Previously automatic tutorial approvals require explicit review after migration;
see [upgrading existing approvals](../nodes/training-plan-security-manager.md#upgrading-existing-approvals)
before upgrading a node database.

Note that due to the two-sided nature of the communication, every change to the messaging protocol
is equivalent to a major change.

!!! warning "Messaging protocol incompatible versions"
    In case of version mismatch, the only solution is to upgrade the software to have the same version on researcher
    and all nodes.
