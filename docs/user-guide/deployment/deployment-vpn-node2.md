# Fed-BioMed deployment with VPN/containers and two node instances on the same node machine

Most real-life deployments require protecting node data. [Deployment using VPN/containers](./deployment-vpn.md) contributes to this goal by providing isolation of the Fed-BioMed instance from third parties.

Deploying two nodes instance on the same node machine is not the normal real life VPN/containers deployment scenario, which consists of one node instance per node machine.

Nevertheless, this scenario can be useful for testing purpose. For example, secure aggregation requires at least 2 nodes participating in the experiment. This scenario allows testing secure aggregation with VPN/containers while all components (researcher + 2 nodes) are running on the same machine.

Operating a second node instance on the same node machine is mostly equivalent to operating the first node instance. Commands are adapted by replacing any occurrence of:

- *node* by **node2**
- *gui* by **gui2**
- *NODETAG* by **NODE2TAG**
- *MPSPDZ_PORT=14001* by **MPSPDZ_PORT=14002**
- *`https://localhost:8443`* by **`https://localhost:8444`**


## Deploy a second node instance on the same node machine

This part of the tutorial is executed once on each node that runs a second node instance, after deploying the server.
It covers the initial deployment, including build, configuration and launch of containers.

Some commands are executed on the node side, while some commands are executed on the server side (pay attention to the prompt).

For each node, choose a **unique** node tag (eg: *NODE2TAG* in this example) that represents this specific node instance for server side management commands.

* build node-side containers

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn build node2 gui2
    ```

* **on the server side**, generate a configuration for this node (known as *NODE2TAG*)

    ```bash
    [user@server $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
    [user@server $] docker compose exec vpnserver bash -ci 'python ./vpn/bin/configure_peer.py genconf node NODE2TAG'
    ```

    The configuration file is now available on the server side in path `${FEDBIOMED_DIR}/envs/vpn/docker/vpnserver/run_mounts/config/config_peers/node/NODE2TAG/config.env` or with command :

    ```bash
    [user@server $] docker compose exec vpnserver cat /config/config_peers/node/NODE2TAG/config.env
    ```

* copy the configuration file from the server side **to the node side** via a secure channel, to path `/tmp/config2.env` on the node.

    In most real life deployments, one shouldn't have access to both server side and node side. Secure channel in an out-of-band secured exchange (outside of Fed-BioMed scope) between the server administrator and the node administrator that provides mutual authentication of the parties, integrity and privacy of the exchanged file.

    In a test deployment, one may be connected both on server side and node side. In this case, you just need to cut-paste or copy the file to the node.

    Use the node's copy of the configuration file:

    ```bash
    [user@node $] cp /tmp/config2.env ./node2/run_mounts/config/config.env
    ```

* start `node2` container

    ```bash
    [user@node $] docker compose up -d node2
    ```

* retrieve the `node2`'s publickey

    ```bash
    [user@node $] docker compose exec node2 wg show wg0 public-key | tr -d '\r' >/tmp/publickey2-nodeside
    ```

* copy the public key from the node side **to the server side** via a secure channel (see above), to path `/tmp/publickey2-serverside` on the server.

* **on the server side** finalize configuration of the VPN keys for this node (known as *NODE2TAG*)

    ```bash
    [user@server $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
    [user@server $] docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py add node NODE2TAG $(cat /tmp/publickey2-serverside)"
    ```

* check containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node2
    ```

    `node2` container should be up and able to ping the VPN server

    ```bash
    ** Checking docker VPN images & VPN access: node2
    - container node2 is running
    - pinging VPN server from container node2 -> OK
    ```

    `node2` container is now ready to be used.

* **optionally** force the use of secure aggregation by the node (node will refuse to train without the use of secure aggregation):

    ```bash
    [user@node $] export FORCE_SECURE_AGGREGATION=True
    ```

* do initial node configuration

    ```bash
    [user@node $] docker compose exec -u $(id -u) node2 bash -ci 'export FORCE_SECURE_AGGREGATION='${FORCE_SECURE_AGGREGATION}'&& export MPSPDZ_IP=$VPN_IP && export MPSPDZ_PORT=14002 && export MQTT_BROKER=10.220.0.2 && export MQTT_BROKER_PORT=1883 && export UPLOADS_URL="http://10.220.0.3:8000/upload/" && export PYTHONPATH=/fedbiomed && export FEDBIOMED_NO_RESET=1 && eval "$(conda shell.bash hook)" && conda activate fedbiomed-node && ENABLE_TRAINING_PLAN_APPROVAL=True ALLOW_DEFAULT_TRAINING_PLANS=True ./scripts/fedbiomed_run node configuration create'
    ```


Optionally launch the node GUI :

* start `gui2` container

    ```bash
    [user@node $] docker compose up -d gui2
    ```

* check containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node2 gui2
    ```

    Node side containers should be up and able to ping the VPN server

    ```bash
    ** Checking docker VPN images & VPN access: node2 gui2
    - container node2 is running
    - container gui2 is running
    - pinging VPN server from container node2 -> OK
    ```

    `node2` and `gui2` containers are now ready to be used.


## Use the second node instance on the same node machine

This part is executed at least once on each node that runs a second node instance, after deploying the node side containers.

Setup the node by sharing datasets and by launching the Fed-BioMed node:

* if node GUI is launched, it can be used to share datasets. On the node side machine, connect to `http://localhost:8485`

* connect to the `node2` container and launch commands, for example :

    * connect to the container

        ```bash
        [user@node $] docker compose exec -u $(id -u) node2 bash -ci 'export MPSPDZ_IP=$VPN_IP && export MPSPDZ_PORT=14002 && export MQTT_BROKER=10.220.0.2 && export MQTT_BROKER_PORT=1883 && export UPLOADS_URL="http://10.220.0.3:8000/upload/" && export PYTHONPATH=/fedbiomed && export FEDBIOMED_NO_RESET=1 && eval "$(conda shell.bash hook)" && conda activate fedbiomed-node && bash'
        ```

    * start the Fed-BioMed node, for example in background:

        ```bash
        [user@node2-container $] nohup ./scripts/fedbiomed_run node start >./fedbiomed_node.out &
        ```

    * share one or more datasets, for example a MNIST dataset or an interactively defined dataset (can also be done via the GUI):

        ```bash
        [user@node2-container $] ./scripts/fedbiomed_run node -am /data
        [user@node2-container $] ./scripts/fedbiomed_run node add
        ```

Example of a few more possible commands:

* optionally list shared datasets:

    ```bash
    [user@node2-container $] ./scripts/fedbiomed_run node list
    ```        

* optionally register a new [authorized training plan](../../tutorials/security/training-with-approved-training-plans.ipynb) previously copied on the node side in `${FEDBIOMED_DIR}/envs/vpn/docker/node2/run_mounts/data/my_training_plan.txt`

    ```bash
    [user@node2-container $] ./scripts/fedbiomed_run node --register-training-plan
    ```
    Indicate `/data/my_training_plan.txt` as path of the training plan file.


## Configure secure aggregation for the server side, for a second node instance

This part of the tutorial is optionally executed once on each node that runs a second node instance.
It is necessary before this component can use secure aggregation in an experiment.

* reuse the `/tmp/cert-secagg` created on this node for the first node instance

    Make the server secagg certificate available to the node:

    ```bash
    [user@node $] cp /tmp/cert-secagg ./node2/run_mounts/etc/cert-secagg
    ```

* [connect to the `node2` container](#use-the-second-node-instance-on-the-same-node-machine) using the command line

    * register the researcher certificate using the **researcher ID, IP address and port as indicated in the server registration instructions**, in this example

        ```bash
        [user@node2-container $] # this is an example, please cut-paste from your registration instructions
        [user@node2-container $] # ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi researcher_2bd34852-830b-48f0-9f58-613f3e643d42  --ip 10.222.0.2 --port 14000
        ```

    * check the deployed certificate

        ```bash
        [user@node2-container $] ./scripts/fedbiomed_run node certificate list
        ```


## Configure secure aggregation for the node side, for a second node instance

This part of the tutorial is optionally executed once on each node that runs a second node instance.
It is necessary before this component can use secure aggregation in an experiment.

* reuse the `/tmp/cert-secagg` created on this node for the first node instance

    Make the node secagg certificate available to the other node:

    ```bash
    [user@othernode $] cp /tmp/cert-secagg ./node2/run_mounts/etc/cert-secagg
    ```

* [connect to the `node2` container](#use-the-second-node-instance-on-the-same-node-machine) using the command line

    * register the node certificate using the **node ID, IP address and port as indicated in the node registration instructions**, in this example

        ```bash
        [user@othernode2-container $] # this is an example, please cut-paste from your registration instructions
        [user@othernode2-container $] # ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg -pi node_964bdca9-809d-49b8-a9c4-8ba3d108c1ae  --ip 10.221.0.2 --port 14001
        ```

    * check the deployed certificate

        ```bash
        [user@othernode2-container $] ./scripts/fedbiomed_run node certificate list
        ```
