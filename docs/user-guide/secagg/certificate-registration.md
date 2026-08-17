
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
MIIDZTCCAk2gAwIBAgIUcZUY1BichrLgNLG5gUGWKTUo9XUwDQYJKoZIhvcNAQEL
BQAwOjE4MDYGA1UEAwwvUkVTRUFSQ0hFUl9lMWM1YzEwMS0wYzc5LTQzYjEtOGJm
MS1kNzBiMDZiOTE4MzAwHhcNMjYwODE3MTQxMTUzWhcNMzEwODE2MTQxMTUzWjA6
MTgwNgYDVQQDDC9SRVNFQVJDSEVSX2UxYzVjMTAxLTBjNzktNDNiMS04YmYxLWQ3
MGIwNmI5MTgzMDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMaMgorg
nD7g2PHYI9pN64CCjAfOkGu7duAChq/wESvNZ5gZDL+tv4ud8OlG/6Ia13BxAC4f
7cNM/LOwTP0I6tdFhbLwNIusKg3y9jBvstIR6d7FZ9lzC0TLa5LFSqB0sXS1KCer
ghQvVOUgzfp+gTxQrdV84ioSJKbVT5gIxqTpBlorbUDkg7JrDn/w6SoKORz8PG9m
a4AGrrVP7ZEESpviuBd+0rdUsRPNg/0GRzEJEqJgugCLLiHgGtVbK9f7CmEeVd8O
9gUWFIGbm/vd1fpTEpDmcRXoluX0YE70jBrOiil9Of3CE8vHjRFJrjhb1JwmRFal
ol+olXQ3zML6/k8CAwEAAaNjMGEwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwEwLAYDVR0RBCUwI4IQZmJtLmhvc3BpdGFs
Lm9yZ4IJbG9jYWxob3N0hwR/AAABMA0GCSqGSIb3DQEBCwUAA4IBAQARVtpjrdeA
VU5yGFR6rQvyHX+XzNHfFC0DJLutu22rXHOIzcRYT0DZy9YEO/IuIBOE2dTJXulB
kGt5DnR5LmLpQgciaxHtVWfITARB+EkQzYaD3fMIkDROXh5bcOAN0eUI0xdJ/xQU
1OIN1qLEmSnFd1AFBwwT6DiSsnxMU7Ac+nvCspuYTY/hNBo62bLGiHYJfL4nAlsw
ifHDzTyPz5/N+cAHUF1yMYZUq08ltFtvQ5TVH8bECCaFoqc1A3zgHHPJ2BYVw40F
Aww/WF5kr2kFU0PiQ0dRqhh+cmg2YTcP4zzTWi2balKATVWUO7MaesN8pHgZ+LDp
/XPkzA+YM1cF
-----END CERTIFICATE-----

Please follow the instructions below to register this certificate:


 1- Copy certificate content into a file e.g 'Hospital1.pem'
 2- On each node, change your directory to 'fedbiomed' root
 3- Run: fedbiomed node certificate register -pk [PATH WHERE CERTIFICATE IS SAVED]

The party id (RESEARCHER_e1c5c101-0c79-43b1-8bf1-d70b06b91830) is read from the certificate, so `-pi` is not needed.
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
