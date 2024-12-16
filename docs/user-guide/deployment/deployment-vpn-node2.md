# Fed-BioMed deployment with VPN/containers and two node instances on the same node machine

Most real-life deployments require protecting node data. [Deployment using VPN/containers](./deployment-vpn.md) contributes to this goal by providing isolation of the Fed-BioMed instance from third parties.

Deploying two nodes instance on the same node machine is not the normal real life VPN/containers deployment scenario, which consists of one node instance per node machine.

Nevertheless, this scenario can be useful for testing purpose. For example, secure aggregation requires at least 2 nodes participating in the experiment. This scenario allows testing secure aggregation with VPN/containers while all components (researcher + 2 nodes) are running on the same machine.

Operating a second node instance on the same node machine is mostly equivalent to operating the first node instance. Commands are adapted by replacing any occurrence of:

- *node* by **node2**
- *gui* by **gui2**
- *NODETAG* by **NODE2TAG**
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
    [user@node $] export FBM_SECURITY_FORCE_SECURE_AGGREGATION=True
    ```

* do initial node configuration

    ```bash
    [user@node $] docker compose exec -u $(id -u) node2 bash -ci 'export FBM_SECURITY_FORCE_SECURE_AGGREGATION='${FBM_SECURITY_FORCE_SECURE_AGGREGATION}' && export FBM_RESEARCHER_IP=10.222.0.2 && export FBM_RESEARCHER_PORT=50051 && export PYTHONPATH=/fedbiomed && FBM_SECURITY_TRAINING_PLAN_APPROVAL=True FBM_SECURITY_ALLOW_DEFAULT_TRAINING_PLANS=True ./scripts/fedbiomed_run environ-node configuration create --component NODE --use-current'
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
        [user@node $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
        [user@node $] docker compose exec -u $(id -u) node2 bash -ci 'export PYTHONPATH=/fedbiomed && eval "$(conda shell.bash hook)" && conda activate fedbiomed-node && bash'
        ```

    * start the Fed-BioMed node, for example in background:

        ```bash
        [user@node2-container $] nohup ./scripts/fedbiomed_run node start >./fedbiomed_node.out &
        ```

    * share one or more datasets, for example a MNIST dataset or an interactively defined dataset (can also be done via the GUI):

        ```bash
        [user@node2-container $] ./scripts/fedbiomed_run node dataset add -m /data
        [user@node2-container $] ./scripts/fedbiomed_run node dataset add
        ```

Example of a few more possible commands:

* optionally list shared datasets:

    ```bash
    [user@node2-container $] ./scripts/fedbiomed_run node dataset list
    ```

* optionally register a new [authorized training plan](../../tutorials/security/training-with-approved-training-plans.ipynb) previously copied on the node side in `${FEDBIOMED_DIR}/envs/vpn/docker/node2/run_mounts/data/my_training_plan.txt`

    ```bash
    [user@node2-container $] ./scripts/fedbiomed_run node training-plan register
    ```
    Indicate `/data/my_training_plan.txt` as path of the training plan file.
