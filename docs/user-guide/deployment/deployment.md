# Fed-BioMed Deployment

Fed-BioMed can be deployed across a range of scenarios, from a single laptop for local testing to a distributed multi-institution federation. This page gives an overview of the available deployment modes and points to the relevant documentation for each.

## Deployment Modes

There are two primary deployment modes, differentiated by whether network communications between components are authenticated and encrypted via a VPN.

### Plain Docker Images

Fed-BioMed provides ready-to-use Docker images for each component (`fedbiomed/node`, `fedbiomed/researcher`). These images start components with minimal configuration and are suitable for:

- **Testing and development** — quickly spin up a local federated learning setup without manual dependency management
- **Customized deployments** — use the provided images as a base and extend them with additional packages, custom datasets, or organisation-specific configuration (see [Extending Docker Images](./docker.md#extending-docker-images))
- **Integration into larger systems** — embed Fed-BioMed nodes or researchers inside an existing infrastructure by building on top of the base images

Plain Docker images do **not** provide VPN-based authentication between components. They are best suited for deployments on trusted networks or for testing purposes.

See the **[Docker deployment guide](./docker.md)** for full instructions.

### Docker Images with VPN

For production deployments where data privacy and network security are required, Fed-BioMed provides a separate set of images that tunnel all inter-component communications through a **WireGuard VPN** with mutual endpoint authentication. This ensures that:

- Only authenticated nodes can join the federation
- All traffic between the researcher and nodes is encrypted
- Node data is protected even when components run on different machines across untrusted networks

This is the recommended mode for real-life deployments involving sensitive data or multiple data provider sites.

See the **[VPN deployment guide](./deployment-vpn.md)** for full instructions.

---

## Choosing a Scenario

| Scenario | Mode | Use Case |
|---|---|---|
| Single machine, no containers | — | [Basic installation](../../getting-started/installation.md): newcomers, local development, contributing to Fed-BioMed |
| Single or multi-machine, plain Docker | Plain Docker | Testing, custom deployments, extending images, trusted networks |
| Single or multi-machine, Docker + VPN | VPN Docker | Production deployments, sensitive data, multi-institution federations |

For multi-machine setups without containers, components must be connected through a highly secure network. See the [README](https://github.com/fedbiomed/fedbiomed/blob/master/README.md#run-the-node-part) for a brief description.

Refer to the [security model](./security-model.md) and [network communications](./matrix.md) documentation to understand which scenario best fits your requirements.
