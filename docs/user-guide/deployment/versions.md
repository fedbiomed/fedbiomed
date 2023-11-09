# Versions

Fed-BioMed stores and checks version numbers for several of its components. 
The [semantics of the versions](https://semver.org/) are as follows:

- different **major** version: incompatibility that results in halting the execution immediately
- different **minor** or **micro** version: backward compatibility, provide a warning message if versions are different

This page tracks the version changes for each component, to provide further information when incompatibilities are
detected.

## Configuration files

### Researcher 

| Version | Changelog                                                                |
|-----|--------------------------------------------------------------------------|
| 0   | Default version assigned prior to the introduction of versioning         |
| 1   | Introduce default/version field tracking the version of this config file |

### Node config file

| Version | Changelog                                                                |
|-----|--------------------------------------------------------------------------|
| 0   | Default version assigned prior to the introduction of versioning         |
| 1   | Introduce default/version field tracking the version of this config file |

### Node state

Node state enable the saving of `Optimizer` and other components.

## Breakpoints

| Version | Changelog                                                                                                 |
|-----|-----------------------------------------------------------------------------------------------------------|
| 0   | Default version assigned prior to the introduction of versioning                                          |
| 1   | Introduce `version` field in breakpoint.json file. In case of incompatible version, see the section below |

## Messaging protocol

Note that due to the two-sided nature of the communication, every change to the messaging protocol
is equivalent to a major change.

!!! warning "Incompatible versions"
    In case of version mismatch, the only solution is to upgrade the software to have the same version on researcher
    and all nodes.

| Version | Changelog                                                                                                 |
|-----|-----------------------------------------------------------------------------------------------------------|
| 0   | Default version assigned prior to the introduction of versioning                                          |
| 1   | Introduce `protocol_version` field in messages. In case of incompatibility see the warning message above. |

