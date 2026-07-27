
# DEPRECATED: Joye-Libert Registration of Certificate and Network Parameters of FL Parties

If Joye-Libert secagg scheme is activated, Fed-BioMed uses the MP-SPDZ library for conducting multi-party computation and also uses the MP-SPDZ network infrastructure for MPC, where each party runs an MP-SPDZ instance listening on an IP and port. In order to proceed with Multi-Party Computation (MPC) in federated experiments, each participating party is required to register the network parameters
of all the other participating parties. The registration must be done before the experiment.

!!! warning "Attention"
    Certificate registration and IP configuration are necessary only for Joye-Libert secure aggregation scheme.


## Registration Through CLI

Fed-BioMed CLI provide options to manage network parameters registration of other parties easily. Registration process
involves two steps as:

1. Getting the certificate and network details of current component and send it to other parties.
2. Registering certificates and network details received from other parties.

### Retrieve certificate and registration instructions

The option `registration-instructions` facilitates the certificate registration process by providing the necessary
details and commands that must be executed by other parties to register the component.

The command below generates registration instructions to assist in the process of registering the researcher
component in the participating parties. This command must be executed by the researcher component, and the instructions sent to other parties for registration.

```shell
fedbiomed researcher certificate registration-instructions
```

the output is:

```
Hi There!


Please find following certificate to register

-----BEGIN CERTIFICATE-----
MIIDBzCCAe+gAwIBAgIDAPQiMA0GCSqGSIb3DQEBCwUAMEYxCjAIBgNVBAMMASox
ODA2BgNVBAoML3Jlc2VhcmNoZXJfZTFjNWMxMDEtMGM3OS00M2IxLThiZjEtZDcw
YjA2YjkxODMwMB4XDTIzMDQwNDEwMTcxN1oXDTI4MDQwMjEwMTcxN1owRjEKMAgG
A1UEAwwBKjE4MDYGA1UECgwvcmVzZWFyY2hlcl9lMWM1YzEwMS0wYzc5LTQzYjEt
OGJmMS1kNzBiMDZiOTE4MzAwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIB
AQC+NQU1HzoNJFWguQY8W97oNWWpkZOtXQE/C63JStZoepbos5nsHpMTZ67Qihfu
BdCe7XNBaZwzTxO3xjKByWocnw+UaadSvNK5zZZNqGiAj3P9F2q1duaWXMldtK/Y
l9bRAW6pp4ri/lnAU5gADDcV7M79pVxhfhMI3xKFP03CA0OqQnXABgZheMCWmtll
x8DVEsKj4jCZSaUqMUHDpxX3l1eUPeDryG3kpcWT28dElBSAynRQznq3StTNghC8
NPMWUQR8uU5HG13n9Xv8+TBZ33b4iXE5Ei24IleFeTJG0PjtRGY6KvEkFKxGvqYs
oKAwpc7u5v0QeDjDeNDrSUhJAgMBAAEwDQYJKoZIhvcNAQELBQADggEBAB5WoUo2
q4VSExJoIpIDEwCimcEKz/pHX9IYBgLGluzGUPfFfN+cjUsmKjzXtIqTRau+LtVO
V/TZ7jRbhTZ7A3FZDrmsE/FOENjUQjFeHIW1Ombqso8BmBfgmn84UF/i1q9rieqZ
jMd+0WppJGp0JNV33mV+veuVbZFaFadRznQ/yUflBcYp0Hfji9/ZU74ivaTdl6vF
LlSIEKmPyHGx+dHub4uzUyfAHlCTxsaOaZzhc8BCR+qbJ499WvKIO5x02r5+mwqN
Ie5FpFt8M14gC+YEfE/KRSOsRlhKHE+wThdNqEC9UpePpkHdS1/9vNs3ql+PojI8
ojZqtVij//Fp8S4=
-----END CERTIFICATE-----

Please follow the instructions below to register this certificate:


 1- Copy certificate content into a file e.g 'Hospital1.pem'
 2- On each node, change your directory to 'fedbiomed' root
 3- Run: fedbiomed node certificate register -pk [PATH WHERE CERTIFICATE IS SAVED]

The party id (researcher_e1c5c101-0c79-43b1-8bf1-d70b06b91830) is read from the certificate, so `-pi` is not needed.
```

The instructions name the component the certificate must be registered on: a researcher
certificate is registered on the nodes, and a node certificate on the researcher. The
certificate carries the party id, so only its file path (`-pk`) is required.


!!! note "Certificates should be shared outside Fed-BioMed through a trusted channel."
    Fed-BioMed does not provide a way to exchange certificate and network parameters internally. Therefore, parameters
    should be shared using third party trusted channels such as e-mail or other messaging channels.

### Registering the certificate

Certificates of other parties must be copied and saved in a file. Then, the file path is given with the option `-pk`.

```shell
fedbiomed [node | researcher] certificate register -pk <certificate-file-path>
```

One of `[node | researcher]` must be chosen according to component type that registers the certificate. A component
registers the certificates of the parties it communicates with, never of its own type: node certificates are registered
on the researcher, and the researcher certificate on each node.

## Registration Through GUI

Currently, certificate registration is not supported through GUI.


## Certificate registration in development/testing mode

Certificate registration is a lengthy procedure, as every network parameter must be registered by every other
participating component. This process can be time-consuming when components are launched locally for
testing or development purposes.

However, Fed-BioMed CLI provides a magic script when all components run in development mode in the same clone. The script parses every configuration
file created in the `etc` directory and registers all available parties automatically in every component.

After all the components are created, please run the following command to complete certificate registration for development
environment.

```shell
fedbiomed certificate-dev-setup
```

!!! warning "Important"
    Secure aggregation setup requires at least 2 nodes and 1 researcher in the FL experiment.
