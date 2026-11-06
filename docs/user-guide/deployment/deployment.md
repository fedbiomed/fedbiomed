# Fed-BioMed Deployment

Fed-BioMed can be deployed across a range of scenarios, from a single laptop for local testing to a distributed multi-institution federation. This page gives an overview of the available deployment modes and points to the relevant documentation for each.

## Deployment Modes

Fed-BioMed components run either directly on a machine or in Docker containers.

Communications between components are encrypted with TLS. [Mutual authentication (mTLS)](./mutual-tls.md) makes the researcher and the nodes authenticate each other at the gRPC level, so only registered nodes can join the federation. It is off by default and has to be enabled on every component of a deployment; leave it off only for local experimentation.

### Docker Images

Fed-BioMed provides ready-to-use Docker images for each component (`fedbiomed/node`, `fedbiomed/researcher`). These images start components with minimal configuration and are suitable for:

- **Testing and development** — quickly spin up a local federated learning setup without manual dependency management
- **Customized deployments** — use the provided images as a base and extend them with additional packages, custom datasets, or organisation-specific configuration (see [Extending Docker Images](./docker.md#extending-docker-images))
- **Integration into larger systems** — embed Fed-BioMed nodes or researchers inside an existing infrastructure by building on top of the base images

See the **[Docker deployment guide](./docker.md)** for full instructions.

---

## Choosing a Scenario

| Scenario | Use Case |
|---|---|
| Single machine, no containers | [Basic installation](../../getting-started/installation.md): newcomers, local development, contributing to Fed-BioMed |
| Single or multi-machine, Docker | Testing, extending images |
| Single or multi-machine, Docker + mTLS | Production deployments, sensitive data, multi-institution federations |

For multi-machine setups without containers, point each node to the researcher address (see [node environment variables](../nodes/configuring-nodes.md#environment-variables)) and [enable mutual authentication](./mutual-tls.md#enabling-mutual-authentication) on every component.

Refer to the [security model](./security-model.md) and [network communications](./matrix.md) documentation to understand which scenario best fits your requirements.

!!! note "VPN"
    Fed-BioMed versions earlier than 6.5 relied on a VPN to protect communications between components. Starting from version 6.5, authentication is based on [mutual authentication (mTLS)](./mutual-tls.md) and no VPN is required. To optionally implement a VPN as an additional layer of protection, see [Running Docker Images Behind a WireGuard VPN](./docker-vpn.md).
