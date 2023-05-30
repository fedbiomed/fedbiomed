# Fed-BioMed deployment on multiple machines with VPN/containers using secure aggregation

Most real-life deployments require protecting node data. [Deployment using VPN/containers](./deployment-vpn.md) contributes to this goal by providing isolation of the Fed-BioMed instance from third parties.

Using secure aggregation for protecting model parameters from nodes' local training adds another layer of security.

## Configure secure aggregation for the server side

This part of the tutorial is optionally executed once for the server.
It is necessary before this component can use secure aggregation in an experiment.

* [connect to the `researcher` container](./deployment-vpn.md#use-the-server) using the command line

* **on the server side**, generate the secagg setup instructions for the server

    ```bash
    [user@server-container $] ./scripts/fedbiomed_run researcher certificate registration-instructions
    ```

    Output contains a certificate of a public key for the server and instructions for registration:

    ```bash
    Hi There! 


    Please find following certificate to register 

    -----BEGIN CERTIFICATE-----
    MIIDBzCCAe+gAwIBAgIDASvKMA0GCSqGSIb3DQEBCwUAMEYxCjAIBgNVBAMMASox
    ...
    qaX4EJXbAjS50P8=
    -----END CERTIFICATE-----

    Please follow the instructions below to register this certificate:


     1- Copy certificate content into a file e.g 'Hospital1.pem'
     2- Change your directory to 'fedbiomed' root
     2- Run: "scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE CERTIFICATE IS SAVED] -pi researcher_2bd34852-830b-48f0-9f58-613f3e643d42  --ip 10.222.0.2 --port 14000"
        Examples commands to use for VPN/docker mode:
          ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi researcher_2bd34852-830b-48f0-9f58-613f3e643d42 --ip 10.222.0.2 --port 14000
          ./scripts/fedbiomed_run researcher certificate register -pk ./etc/cert-secagg -pi researcher_2bd34852-830b-48f0-9f58-613f3e643d42 --ip 10.222.0.2 --port 14000
    ```

* then **repeat for each node** the propagation of the server's secagg configuration

    * transmit the registration instructions from the server side **to the node side** via a secure channel.

        In most real life deployments, one shouldn't have access to both server side and node side. Secure channel in an out-of-band secured exchange (outside of Fed-BioMed scope) between the server administrator and the node administrator that provides mutual authentication of the parties, integrity and privacy of the exchanged file.

        In a test deployment, one may be connected both on server side and node side. In this case, you just need to cut-paste or save to a file on the node.

        Copy certificate content to file `/tmp/cert-secagg` on the node.

        Make the server secagg certificate available to the node:

        ```bash
        [user@node $] cp /tmp/cert-secagg ./node/run_mounts/etc/cert-secagg
        ```

    * [connect to the `node` container](./deployment-vpn.md#use-the-node) using the command line

    * register the server certificate using the **researcher ID, IP address and port as indicated in the server registration instructions**, in this example

        ```bash
        [user@node-container $] # this is an example, please cut-paste from your registration instructions
        [user@node-container $] # ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi researcher_2bd34852-830b-48f0-9f58-613f3e643d42  --ip 10.222.0.2 --port 14000
        ```

    * check the deployed certificate

        ```bash
        [user@node-container $] ./scripts/fedbiomed_run node certificate list
        ```

    * optionally [propagate to a second node instance on the same node](./deployment-vpn-node2.md#configure-secure-aggregation-for-the-server-side-for-a-second-node-instance), if a second node instance was previously deployed


## Configure secure aggregation for the node side

This part of the tutorial is optionally executed once for each node instance.
It is necessary before this component can use secure aggregation in an experiment.

* [connect to the `node` container](./deployment-vpn.md#use-the-node) using the command line

* **on the node side**, generate the secagg setup instructions for the node (**if running a second node instance on the node, repeat for that instance**)

    ```bash
    [user@node-container $] ./scripts/fedbiomed_run node certificate registration-instructions
    ```

    Output contains a certificate of a public key for the node and instructions for registration:

    ```bash
    Hi There!


    Please find following certificate to register

    -----BEGIN CERTIFICATE-----
    MIIC+jCCAeKgAwIBAgICZRQwDQYJKoZIhvcNAQELBQAwQDEKMAgGA1UEAwwBKjEy
    ...
    fG6KEo0KGnAKgmFpZxtftCBmAiLvZgvJ3LIfMbysfMXy1UFcnvVwoCQznqn1YQ==
    -----END CERTIFICATE-----

    Please follow the instructions below to register this certificate:


     1- Copy certificate content into a file e.g 'Hospital1.pem'
     2- Change your directory to 'fedbiomed' root
     2- Run: "scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE CERTIFICATE IS SAVED] -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae  --ip 10.221.0.2 --port 14001"
        Examples commands to use for VPN/docker mode:
          ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae --ip 10.221.0.2 --port 14001
          ./scripts/fedbiomed_run researcher certificate register -pk ./etc/cert-secagg -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae --ip 10.221.0.2 --port 14001
    ```

* then **repeat for each other node** the propagation of the node's secagg configuration

    * transmit the registration instructions from the node side **to the other node side** via a secure channel.

        In most real life deployments, one shouldn't have access to both nodes sides. Secure channel in an out-of-band secured exchange (outside of Fed-BioMed scope) between the nodes administrators that provides mutual authentication of the parties, integrity and privacy of the exchanged file.

        In a test deployment, one may be connected both nodes sides. In this case, you just need to cut-paste or save to a file on the other node.

        Copy certificate content to file `/tmp/cert-secagg` on the other node.

        Make the node secagg certificate available to the other node:

        ```bash
        [user@othernode $] cp /tmp/cert-secagg ./node/run_mounts/etc/cert-secagg
        ```

    * [connect to the other node's `node` container](./deployment-vpn.md#use-the-node) using the command line

    * register the node certificate using the **node ID, IP address and port as indicated in the node registration instructions**, in this example

        ```bash
        [user@othernode-container $] # this is an example, please cut-paste from your registration instructions
        [user@othernode-container $] # ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae  --ip 10.221.0.2 --port 14001
        ```

    * check the deployed certificate

        ```bash
        [user@othernode-container $] ./scripts/fedbiomed_run node certificate list
        ```

    * optionally [propagate to a second node instance on the other node](./deployment-vpn-node2.md#configure-secure-aggregation-for-the-node-side-for-a-second-node-instance), if a second node instance was previously deployed

* then **propagate to the server** the node's secagg configuration

    * transmit the registration instructions from the node side **to the server side** via a secure channel.

        In most real life deployments, one shouldn't have access to both server side and node side. Secure channel in an out-of-band secured exchange (outside of Fed-BioMed scope) between the server administrator and the node administrator that provides mutual authentication of the parties, integrity and privacy of the exchanged file.

        In a test deployment, one may be connected both on server side and node side. In this case, you just need to cut-paste or save to a file on the server.

        Copy certificate content to file `/tmp/cert-secagg` on the server.

        Make the node secagg certificate available to the node:

        ```bash
        [user@server $] cp /tmp/cert-secagg ./researcher/run_mounts/etc/cert-secagg
        ```

    * [connect to the `researcher` container](./deployment-vpn.md#use-the-server) using the command line

    * register the node certificate using the **node ID, IP address and port as indicated in the node registration instructions**, in this example

        ```bash
        [user@server-container $] # this is an example, please cut-paste from your registration instructions
        [user@server-container $] # ./scripts/fedbiomed_run researcher certificate register -pk ./etc/cert-secagg -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae  --ip 10.221.0.2 --port 14001
        ```

    * check the deployed certificate

        ```bash
        [user@researcher-container $] ./scripts/fedbiomed_run researcher certificate list
        ```

