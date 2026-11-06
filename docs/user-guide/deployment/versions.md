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

## Upgrading an existing deployment

When upgrading from 6.4.1 to a release with mutual authentication, complete these
steps before resuming experiments:

1. **Upgrade the researcher and every node together.** The messaging protocol must
   match across the federation; see [version 8](#version-8) below.
2. **Review each component's `etc/config.ini`.** For compatibility, loading an older
   configuration without `[authentication]` section, automatically starts the component  with `mutual_authentication = False` and logs a warning at startup. Upgrading does not enable authentication automatically. **Add the section to the file and set the value to `True` on the researcher and every node in a deployment.**
3. **Check the existing certificates.** A researcher certificate registered on a node
   must have a DNS or IP Subject Alternative Name (SAN). Older certificates without
   these names must be regenerated or replaced before enabling mTLS. Prefer names
   that nodes use to reach the researcher; see [names on the researcher certificate](./mutual-tls.md#names-on-the-researcher-certificate).
   Check expiry as well: expired certificates cannot be registered.
4. **Exchange and register certificates through a trusted channel.** The researcher
   registers every node's certificate; each node registers the researcher's certificate.
   Node-to-node registration is not needed. After regenerating or replacing a
   certificate, redistribute it and update existing registrations with `--upsert`.
   Follow the [registration procedure](./mutual-tls.md#2-register-the-received-certificates).
5. **Restart components to apply the changes.** Configuration changes and changes to
   a component's own certificate/key require a restart. Restart each node after
   changing its registered researcher certificate. The researcher picks up changes
   to registered node certificates on subsequent handshakes without a restart.
6. **Verify authentication before training.** Check `fedbiomed node status` on each
   node: the connection should be `connected`, with mutual authentication `On` and
   identity verification `Yes`. See
   [verifying and troubleshooting](./mutual-tls.md#verifying-and-troubleshooting).

### Migrating from bundled VPN deployments

The bundled VPN images, `envs/vpn` deployment files, and `scripts/fedbiomed_vpn`
helper have been removed. Existing deployments using them must plan a transition
to the [standard Docker images](./docker.md) or a native installation; updating the
source checkout does not migrate running containers or their volumes.

Back up and preserve each component's configuration, database, certificates/private
keys, datasets, and experiment files before changing containers. Map the existing
component directories to the standard images' persistent paths (`/fbm-node` and
`/fbm-researcher`), and update dataset mounts and researcher addresses as needed.
Complete the mTLS checklist above and verify the replacement deployment before
retiring the old containers.

If institutional policy requires a VPN, provide it independently of Fed-BioMed.
The [optional Docker VPN guide](./docker-vpn.md) illustrates a third-party setup;
it does not migrate the old VPN configuration, and mTLS remains required.

## Messaging protocol

### Version 8

Version 8 removes the packaged default node-to-node EC private key. Channel
setup now carries only public material: `ChannelSetupRequest` and `ChannelSetupReply` are in plaintext.

This setup format is incompatible with version 7, which encrypted the initial
exchange with the repository-wide default key. Researchers and all nodes
participating in the same federation must therefore use messaging protocol
version 8 together.
