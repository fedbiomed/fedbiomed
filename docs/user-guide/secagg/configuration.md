# Secure Aggregation Configuration

Secure aggregation is implemented in Fed-BioMed and can be activated or deactivated as an option through the
configuration. Even if secure aggregation is not configured during the initial installation, Fed-BioMed still works as long as the researcher or node component does not activate it. However, if secure aggregation
is activated in an instance, the infrastructure must have been post-configured for secure aggregation beforehand, after the Fed-BioMed instance  is installed.

To configure an instance for secure aggregation, you need to install [MP-SPDZ](./introduction.md#mp-spdz)
for multi-party computation and provide communication parameters, such as IP address, port, and SSL certificate.
While Fed-BioMed provides magic scripts for configuring communication parameters in development mode, where all
instances are launched in the same local environment, these parameters must be manually configured for
other Fed-BioMed deployment cases.


## MP-SPDZ Installation

Fed-BioMed provides pre-compiled MP-SPDZ protocols and scripts for Linux-based operating systems. However,
for Darwin-based operating systems, MP-SPDZ must be compiled from its source code. Fed-BioMed provides a script
that eases this process by distinguishing the operating system and performing the installation.

To install or configure existing pre-compiled MP-SPDZ scripts for a `Node compoent`, please run following command.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_configure_secagg node
```
For researcher component;

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_configure_secagg researcher
```

!!! note "Development environment"
    In a development environment where researcher and nodes are located in the same local environment, it is sufficient to run the installation once equally as node or researcher.

!!! note "Specifying environment"
    Specifying the environment is necessary to make available the required modules for installation. Declaring
    either node or researcher environments does not make any significant difference, except that the node
    instance would not have the researcher environment and the researcher instance would not have the node
    environment in a deployment scenario.

## Creating a component

Usually, a Fed-BioMed instance (either node or researcher) is created once it is started or a dataset is added.
However, when using secure aggregation, creating Fed-BioMed instances before starting the node gives the
opportunity to modify the configuration.

### Using CLI with default option

The Fed-BioMed CLI provides an option to generate or create a component. It creates a configuration file,
a database file and certificates.

The command below creates a component with a configuration file named `config-n1.ini`.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run environ-node configuration create --component node --name config-n1.ini
```

Other node components can be created by replacing `config-n1.in`i` with any desired unique name.


Fed-BioMed currently allows only a single researcher. Therefore, the following command creates a researcher component
with default configuration file name.


```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run configuration create --component researcher
```

### Updating existing configuration

Depending on the version of Fed-BioMed for which the configuration file is created, some variables may not be
available in the current configuration file after upgrading Fed-BioMed. In such cases, it is always better to
back up the old configuration file and create a new one using the CLI. Afterwards, the values of the backed-up
configuration file can be replaced by the default ones that are generated.



## MP-SPDZ and Secure Aggregation Configuration

After the installation and component creation are completed, configuration files of Fed-BioMed instances can be
modified. Here are the sections that can be configured.

```ini
[mpspdz]
private_key = certs/cert_node_e0394bff-4684-4f84-9c84-6c5e3c683dcb/MPSPDZ_certificate.key
public_key = certs/cert_node_e0394bff-4684-4f84-9c84-6c5e3c683dcb/MPSPDZ_certificate.pem
mpspdz_ip = localhost
mpspdz_port = 14000
allow_default_biprimes = True

```

Node configuration file has extra variables under `security` section regarding secure aggregation
configuration as `secure_aggregation` and `force_secure_aggregation`.  These parameters allow node owners to
manage activate, deactivate and force secure aggregation by default.


```ini
[security]
hashing_algorithm = SHA256
allow_default_training_plans = True
training_plan_approval = False
secure_aggregation = True
force_secure_aggregation = False
```

### Activating, Deactivating and Forcing Secure Aggregation

Nodes have the privilege of activating, deactivating, and enforcing secure aggregation. This means that model parameters
can either be encrypted (required), optionally encrypted, or unencrypted. If a node requires model encryption and a
training request from a researcher does not include secure aggregation context, the node will refuse training. If
secure aggregation is allowed but not forced by the node, end-users are able to send training requests
with or without secure aggregation.

In a federated setup, if one of the nodes requires secure aggregation but the researcher does not activate it,
the FL round fails. Please refer to the researcher secure aggregation interface for more details.

!!! note "Researcher"
    Researcher configuration file does not have parameter regarding secure aggregation activation. However,
    secure aggregation context is managed through [Experiment][fedbiomed.researcher.experiment] interface (class).


### MP-SPDZ Network Parameters

#### Certificates

When a component is created, Fed-BioMed CLI automatically generates SSL certificates. These certificates are
self-signed certificates, and they can be re-generated or replaced by certificates signed by third party authorities.
Certificate files should be located in `${FEDBIOMED_DIR}/etc/certs` and path should be relative to `${FEDBIOMED_DIR}`.

Fed-BioMed generates 2048 bits RSA key and signs it using `SHA256`. Generated private keys and certificates located in
`${FEDBIOMED_DIR}/etc/certs/cert_<component_id>`, and named as `MPSPDZ_certificate.key` and `MPSPDZ_certificate.pem`.
`ORGANIZATION_NAME` attribute of the certificate is set to the component id.

#### Re-generating Certificates

Auto-generated certificates expire after 5 years. However, they can be renewed through the Fed-BioMed CLI at
any desired time. If a certificate has already been generated, the command should be executed with the --force or
-f option.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run node --config config-n1.ini certificate generate -f
```
or

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher certificate generate -f
```

!!! note "Important"
    After SSL certificates are re-generated, other parties that participate and common FL experiment should register
    the new certificate. Please see [certificate registration](./certificate-registration.md)

#### MP-SPDZ Port and IP

Each component of Fed-BioMed in a FL experiment must have IP address and port number assigned for MP-SPDZ. IP address
and port can be set through configuration file or environment variables.

```ini
[mpspdz]
private_key = certs/cert_node_e0394bff-4684-4f84-9c84-6c5e3c683dcb/MPSPDZ_certificate.key
public_key = certs/cert_node_e0394bff-4684-4f84-9c84-6c5e3c683dcb/MPSPDZ_certificate.pem
mpspdz_ip = localhost
mpspdz_port = 14000
allow_default_biprimes = True
```

#### Setting IP and PORT through environment variables

Setting `MPSDPZ_IP` and `MPSDPZ_PORT` environment variables can be set dynamically through environment variables while
starting a Fed-BioMed component. This option does not re-write configuration file but uses the environment variable
value during the component lifetime.

```shell
MPSDPZ_IP=<ip-address> MPSDPZ_PORT=<port> ${FEDBIOMED_DIR}/scripts/fedbiomed_run node --config config-n1.ini start
```

or for researcher

```shell
MPSDPZ_IP=<ip-address> MPSDPZ_PORT=<port> ${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher start
```

!!! warning "Important"
    IP and port information of each party must be known by every other parties before using secure aggregation. Therefore, changing them
    without getting them updated on the other parties causes connection problems during the MPC.

#### Auto-increment Port in Dev-Mode

Fed-BioMed configuration automatically assigns `localhost` as MP-SPDZ IP by default. The port is incremented automatically
starting from `14000`. The port number for every other generated component in a single clone of Fed-BioMed equals to
the previously assigned port number plus one.


#### Certificate and IP/PORT Registration of FL Parties

Each party must register the network parameters of every other party in order to establish communication for MPC.
This is because MP-SPDZ has its own communication procedure that differs from the one natively provided by Fed-BioMed.
Registration process involves providing network details such as IP, port, and SSL certificate for establishing secure
connections. Fortunately, Fed-BioMed provides scripts to simplify the registration process. For more details, please
refer to the [certificate registration](./certificate-registration.md) documentation.
