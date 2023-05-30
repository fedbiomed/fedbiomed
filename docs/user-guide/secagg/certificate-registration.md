# Registration of Certificate and Network Parameters of FL Parties

Fed-BioMed relies on the MP-SPDZ library for conducting multi-party computation and also uses the MP-SPDZ network
infrastructure for MPC, where each party runs an MP-SPDZ instance listening on an IP and port. In order to proceed with Multi-Party
Computation (MPC) in federated experiments, each participating party is required to register the network parameters
of all the other participating parties. The registration must be done before the experiment.


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
${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher certificate registration-instructions
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
 2- Change your directory to 'fedbiomed' root
 2- Run: "./scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE CERTIFICATE IS SAVED] -pi researcher_e1c5c101-0c79-43b1-8bf1-d70b06b91830  --ip 193.0.0.1 --port 14002"
    Examples commands to use for VPN/docker mode:
      ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi researcher_e1c5c101-0c79-43b1-8bf1-d70b06b91830 --ip 193.0.0.1 --port 14002
      ./scripts/fedbiomed_run researcher certificate register -pk ./etc/cert-secagg -pi researcher_e1c5c101-0c79-43b1-8bf1-d70b06b91830 --ip 193.0.0.1 --port 14002
```

The aforementioned instructions provide essential details for registering a party among the other participants in a
federated experiment. The information includes the certificate, component identification number (`-pi`), IP address
(`--ip`), and port (`--port`).


!!! note "Certificates should be shared outside Fed-BioMed through a trusted channel."
    Fed-BioMed does not provide a way to exchange certificate and network parameters internally. Therefore, parameters
    should be shared using third party trusted channels such as e-mail or other messaging channels.

### Registering the certificate

Certificates of other parties should be registered with their component ID, IP and port information. Certificates must
be copied and saved in a file. Then, the file path is given with the option `-pk`.

```shell
${FEDBIOMED_DIR}/scripts/fedbiomed_run [node | researcher] certificate register -pk <certificate-file-path> -pi <component-id> --ip  <IP> --port <PORT>"
```

One of `[node | researcher]` must be chosen according to component type that registers the certificate.

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
${FEDBIOMED_DIR}/scripts/fedbiomed_run certificate-dev-setup
```

!!! warning "Important"
    Secure aggregation setup requires at least 2 nodes and 1 researcher in the FL experiment.