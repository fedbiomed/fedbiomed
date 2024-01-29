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

Note that due to the two-sided nature of the communication, every change to the messaging protocol
is equivalent to a major change.

!!! warning "Messaging protocol incompatible versions"
    In case of version mismatch, the only solution is to upgrade the software to have the same version on researcher
    and all nodes.
