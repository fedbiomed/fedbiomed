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

For multi-machine setups without containers, enable mutual authentication on every component. See the [README](https://github.com/fedbiomed/fedbiomed/blob/master/README.md#run-the-node-part) for a brief description.

Refer to the [security model](./security-model.md) and [network communications](./matrix.md) documentation to understand which scenario best fits your requirements.
