# Fed-BioMed deployment on multiple machines with VPN/containers

Most real-life deployments require protecting node data. Deployment using VPN/containers contributes to this goal by providing isolation of the Fed-BioMed instance from third parties. All communications between the components of a Fed-BioMed instance occur inside WireGuard VPN tunnels with mutual authentication of the VPN endpoints. Using containers can also ease installation on multiple sites.

This tutorial details a deployment scenario where:

* Fed-BioMed network and researcher components run on the same machine ("the **server**") in the following `docker` containers
    - `vpnserver` / `fedbiomed/vpn-vpnserver`: WireGuard server
    - `mqtt` / `fedbiomed/vpn-mqtt`: MQTT message broker server
    - `restful` / `fedbiomed/vpn-restful`: HTTP REST communication server
    - `researcher` / `fedbiomed/vpn-researcher`: a researcher jupyter notebooks
* several Fed-BioMed **node** components run, one node per machine with the following containers
    - `node` / `fedbiomed/vpn-node`: a node component
    - `gui` / `fedbiomed/vpn-gui`: a GUI for managing node component data (optional)
* all communications between the components are tunneled through a VPN


## Requirements

!!! info "Supported operating systems and software requirements"
    Supported operating systems for containers/VPN deployment include **Fedora 38**, **Ubuntu 22.04 LTS**. Should also work for most recent Linux, **MacOS X 12.6.6 and 13**, **Windows 11** with WSL2 using Ubuntu-22.04 distribution. Also requires **docker compose >= 2.0**.

    Check here for [detailed requirements](https://github.com/fedbiomed/fedbiomed/blob/master/envs/vpn/README.md#requirements).

!!! info "Account privileges"
    Components deployment requires an account which can use docker (typically belonging to the `docker` group).
    Using a dedicated service account is a good practice.
    No access to the administrative account is needed, usage of `root` account for deploying components is discouraged to follow the principle of privilege minimization.

!!! info "Web proxy"
    On sites where web access uses a proxy you need to [configure web proxy for docker](#proxy).

!!! info "User or group ID for containers"
    By default, Fed-BioMed uses the current account's user and group ID for building and running containers.

    Avoid using low ID for user or group ( < 500 for MacOS, < 1000 for Linux ) inside containers. They often conflict with pre-existing user or group account in the container images. This results in unhandled failures when setting up or starting the containers. Check your account id with `id -a`.

    Use the `CONTAINER_USER`, `CONTAINER_UID`, `CONTAINER_GROUP` and `CONTAINER_GID` variables to use alternate values, eg for MacOS:

    MacOS commonly uses group `staff:20` for user accounts, which conflicts with Fed-BioMed VPN/containers mode. So a good configuration choice for MacOS can be:

    ```bash
    export CONTAINER_GROUP=fedbiomed
    export CONTAINER_GID=1111
    ```


More options for containers/VPN deployment are not covered in this tutorial but [can be found here](https://github.com/fedbiomed/fedbiomed/blob/master/envs/vpn/README.md) including:

* using GPU in `node` container
* building containers (eg: `node` and `gui`) on one machine, using this pre-built containers on the nodes
* using different identity (account) for building and launching a container
* deploying network and researcher on distinct machines


## Notations

In this tutorial we use the following notations:

- `[user@server $]` means the command is launched on the server machine (outside containers)
- `[user@node $]` means the command is launched on a node machine (outside containers)
- for commands typed inside containers, `[root@vpnserver-container #]` means the command is launched inside the `vpnserver` container as root, `[user@node-container $]` means the command is launched inside the `vpnserver` container with same user account as outside the container


## Deploy on the server side


This part of the tutorial is executed once on the server side, before deploying the nodes.
It covers the initial server deployment, including build, configuration and launch of containers.

* download Fed-BioMed software by doing a local clone of the git repository: 

    ```bash
    [user@server $] git clone -b master https://github.com/fedbiomed/fedbiomed.git
    [user@server $] cd fedbiomed
    [user@server $] export FEDBIOMED_DIR=$PWD # use setenv for *csh
    [user@server $] cd envs/vpn/docker
    ```

    For the rest of this tutorial `${FEDBIOMED_DIR}` represents the base directory of the clone.

    `docker compose` commands need to be launched from `${FEDBIOMED_DIR}/envs/vpn/docker directory`.

* clean running containers, containers files, temporary files

    ``` bash
    [user@server $] source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean
    ```

* **optionally** clean the container images to force build fresh new images

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image
    ```

* build server-side containers

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn build vpnserver mqtt restful researcher
    ```

* configure the VPN keys for containers running on the server side, after starting the `vpnserver` container

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn configure mqtt restful researcher
    ```

* start other server side containers

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn start mqtt restful researcher
    ```

* check all containers are running as expected on the server side

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status vpnserver mqtt restful researcher
    ```

    Server side containers should be up and able to ping the VPN server

    ```bash
    ** Checking docker VPN images & VPN access: vpnserver mqtt restful researcher
    - container vpnserver is running
    - container mqtt is running
    - container restful is running
    - container researcher is running
    - pinging VPN server from container vpnserver -> OK
    - pinging VPN server from container mqtt -> OK
    - pinging VPN server from container restful -> OK
    - pinging VPN server from container researcher -> OK
    ```

    Server side containers are now ready for node side deployment.


## Deploy on the node side

This part of the tutorial is executed once on each node, after deploying the server.
It covers the initial deployment, including build, configuration and launch of containers.

Some commands are executed on the node side, while some commands are executed on the server side (pay attention to the prompt).

For each node, choose a **unique** node tag (eg: *NODETAG* in this example) that represents this specific node instance for server side management commands.

* download Fed-BioMed software by doing a local clone of the git repository: 

    ```bash
    [user@node $] git clone -b master https://github.com/fedbiomed/fedbiomed.git
    [user@node $] cd fedbiomed 
    [user@node $] export FEDBIOMED_DIR=$PWD # use setenv for *csh
    [user@node $] cd envs/vpn/docker
    ```

    For the rest of this tutorial `${FEDBIOMED_DIR}` represents the base directory of the clone.

    `docker compose` commands need to be launched from `${FEDBIOMED_DIR}/envs/vpn/docker directory`.

* clean running containers, containers files, temporary files (skip that step if node and server run on the same machine)

    ``` bash
    [user@node $] source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean
    ```

* **optionally** clean the container images to force build fresh new images

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image
    ```

* build node-side containers

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn build node gui
    ```

* **on the server side**, generate a configuration for this node (known as *NODETAG*)

    ```bash
    [user@server $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
    [user@server $] docker compose exec vpnserver bash -ci 'python ./vpn/bin/configure_peer.py genconf node NODETAG'
    ```

    The configuration file is now available on the server side in path `${FEDBIOMED_DIR}/envs/vpn/docker/vpnserver/run_mounts/config/config_peers/node/NODETAG/config.env` or with command :

    ```bash
    [user@server $] docker compose exec vpnserver cat /config/config_peers/node/NODETAG/config.env
    ```

* copy the configuration file from the server side **to the node side** via a secure channel, to path `/tmp/config.env` on the node.

    In most real life deployments, one shouldn't have access to both server side and node side. Secure channel in an out-of-band secured exchange (outside of Fed-BioMed scope) between the server administrator and the node administrator that provides mutual authentication of the parties, integrity and privacy of the exchanged file.

    In a test deployment, one may be connected both on server side and node side. In this case, you just need to cut-paste or copy the file to the node.

    Use the node's copy of the configuration file:

    ```bash
    [user@node $] cp /tmp/config.env ./node/run_mounts/config/config.env
    ```

* start `node` container

    ```bash
    [user@node $] docker compose up -d node
    ```

* retrieve the `node`'s publickey

    ```bash
    [user@node $] docker compose exec node wg show wg0 public-key | tr -d '\r' >/tmp/publickey-nodeside
    ```   

* copy the public key from the node side **to the server side** via a secure channel (see above), to path `/tmp/publickey-serverside` on the server.

* **on the server side** finalize configuration of the VPN keys for this node (known as *NODETAG*)

    ```bash
    [user@server $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
    [user@server $] docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py add node NODETAG $(cat /tmp/publickey-serverside)"
    ```

* check containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node
    ```

    `node` container should be up and able to ping the VPN server

    ```bash
    ** Checking docker VPN images & VPN access: node
    - container node is running
    - pinging VPN server from container node -> OK
    ```

    `node` container is now ready to be used.

* **optionally** force the use of secure aggregation by the node (node will refuse to train without the use of secure aggregation):

    ```bash
    [user@node $] export FORCE_SECURE_AGGREGATION=True
    ```

* do initial node configuration

    ```bash
    [user@node $] docker compose exec -u $(id -u) node bash -ci 'export FORCE_SECURE_AGGREGATION='${FORCE_SECURE_AGGREGATION}'&& export MPSPDZ_IP=$VPN_IP && export MPSPDZ_PORT=14001 && export MQTT_BROKER=10.220.0.2 && export MQTT_BROKER_PORT=1883 && export UPLOADS_URL="http://10.220.0.3:8000/upload/" && export PYTHONPATH=/fedbiomed && export FEDBIOMED_NO_RESET=1 && eval "$(conda shell.bash hook)" && conda activate fedbiomed-node && ENABLE_TRAINING_PLAN_APPROVAL=True ALLOW_DEFAULT_TRAINING_PLANS=True ./scripts/fedbiomed_run node configuration create'
    ```


Optionally launch the node GUI :

* **optionally** authorize connection to node GUI from distant machines. By default, only connection from local machine (`localhost`) is authorized.

    ```bash
    [user@node $] export GUI_SERVER_IP=0.0.0.0
    ```

    To authorize distant connection to only one of the node machine's IP addresses use a command of the form `export GUI_SERVER_IP=a.b.c.d` where `a.b.c.d` is one of the IP addresses of the node machine.

    For security reasons, when authorizing connection from distant machines, it is strongly recommended to **use a custom SSL certificate signed by a well-known authority**.

    !!! note "Custom SSL certificates  for GUI"

        GUI will start serving on port 8443 with self-signed certificates. These certificates will be identified as risky by the 
        browsers, and users will have to approve them. However, it is also possible to set custom trusted SSL certificates by 
        adding `crt` and `key` files to the `${FEDBIOMED_DIR}/envs/vpn/docker/gui/run_mounts/certs` directory before starting the GUI.

        When adding these files, please ensure that:

        - the certificate extension is `.crt` and the key file extension is `.key`
        - there is no more than one file for each certificate and key

* **optionally** restrict the HTTP host names that can be used to connect to the node GUI. By default all the host names (DNS CNAME) of the node machine can be used.

    For example, if the node machine has two host names `my.fqdn.com` and `other.alias.org`, use syntax like `export GUI_SERVER_NAME=my.fqdn.com` or `GUI_SERVER_NAME='*.fqdn.com'` (don't forget the enclosing single quotes) to authorize only requests using the first name (eg: `https://my.fqdn.com`) to reach the node GUI. Use the syntax of Nginx [`server_name`](http://nginx.org/en/docs/http/server_names.html).

* start `gui` container

    ```bash
    [user@node $] docker compose up -d gui
    ```

* check containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node gui
    ```

    Node side containers should be up and able to ping the VPN server

    ```bash
    ** Checking docker VPN images & VPN access: node gui
    - container node is running
    - container gui is running
    - pinging VPN server from container node -> OK
    ```

    `node` and `gui` containers are now ready to be used.


## Optionally deploy a second node instance on the same node

Optionally deploy a second node instance on the same node (useful for testing purpose, not a normal deployment scenario):

* [deploy second node on the same machine](./deployment-vpn-node2.md#deploy-a-second-node-instance-on-the-same-node-machine)

This part of the tutorial is optionally executed on some nodes, after deploying the server.


## Optionally configure secure aggregation

Optionally configure secure aggregation for additional security:

* for [the server side](./deployment-vpn-secagg.md#configure-secure-aggregation-for-the-server-side)
* for [the node side](./deployment-vpn-secagg.md#configure-secure-aggregation-for-the-node-side)

This part of the tutorial is optionally executed once on each node and once on the server.
It is necessary before this component can use secure aggregation in an experiment.


## Use the node

This part is executed at least once on each node, after deploying the node side containers.

Setup the node by sharing datasets and by launching the Fed-BioMed node:

* if node GUI is launched, it can be used to share datasets. On the node side machine, connect to `https://localhost:8443` (or `https://<host_name_and_domain>:8443` if connection from distant machine is authorized)

* connect to the `node` container and launch commands, for example :

    * connect to the container

        ```bash
        [user@node $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
        [user@node $] docker compose exec -u $(id -u) node bash -ci 'export MPSPDZ_IP=$VPN_IP && export MPSPDZ_PORT=14001 && export MQTT_BROKER=10.220.0.2 && export MQTT_BROKER_PORT=1883 && export UPLOADS_URL="http://10.220.0.3:8000/upload/" && export PYTHONPATH=/fedbiomed && export FEDBIOMED_NO_RESET=1 && eval "$(conda shell.bash hook)" && conda activate fedbiomed-node && bash'
        ```

    * start the Fed-BioMed node, for example in background:

        ```bash
        [user@node-container $] nohup ./scripts/fedbiomed_run node start >./fedbiomed_node.out &
        ```

    * share one or more datasets, for example a MNIST dataset or an interactively defined dataset (can also be done via the GUI):

        ```bash
        [user@node-container $] ./scripts/fedbiomed_run node -am /data
        [user@node-container $] ./scripts/fedbiomed_run node add
        ```

Example of a few more possible commands:

* optionally list shared datasets:

    ```bash
    [user@node-container $] ./scripts/fedbiomed_run node list
    ```        

* optionally register a new [authorized training plan](../../tutorials/security/training-with-approved-training-plans.ipynb) previously copied on the node side in `${FEDBIOMED_DIR}/envs/vpn/docker/node/run_mounts/data/my_training_plan.txt`

    ```bash
    [user@node-container $] ./scripts/fedbiomed_run node --register-training-plan
    ```
    Indicate `/data/my_training_plan.txt` as path of the training plan file.

## Optionally use a second node instance on the same node

This optional part is executed at least once on the nodes where a second node instance is deployed, after deploying the second node side containers:

* [use second node on the same machine](./deployment-vpn-node2.md#use-the-second-node-instance-on-the-same-node-machine)


## Use the server

This part is executed at least once on the server after setting up the nodes:

* on the server side machine, connect to `http://localhost:8888`, then choose and run a Jupyter notebook

    * make more notebooks available from the server side machine (eg: `/tmp/my_notebook.ipynb`) by copying them to the `samples` directory

        ```bash
        [user@server $] cp /tmp/my_notebook.ipynb ${FEDBIOMED_DIR}/envs/vpn/docker/researcher/run_mounts/samples/
        ``` 
        The notebook is now available in the Jupyter GUI under the `samples` subdirectory of the Jupyter notebook interface.

* if the notebook uses Tensorboard, it can be viewed
    * either embedded inside the Jupyter notebook as explained in the [Tensorboard documentation](../researcher/tensorboard.md)
    * or by connecting to `http://localhost:6006`


Optionally use the researcher container's command line instead of the Jupyter notebooks:

* connect to the `researcher` container

    ```bash
    [user@server $] cd ${FEDBIOMED_DIR}/envs/vpn/docker
    [user@server $] docker compose exec -u $(id -u) researcher bash -ci 'export MPSPDZ_IP=$VPN_IP && export MPSPDZ_PORT=14000 && export MQTT_BROKER=10.220.0.2 && export MQTT_BROKER_PORT=1883 && export UPLOADS_URL="http://10.220.0.3:8000/upload/" && export PYTHONPATH=/fedbiomed && export FEDBIOMED_NO_RESET=1 && eval "$(conda shell.bash hook)" && conda activate fedbiomed-researcher && bash'
    ```

* launch a command, for example a training:

    ```bash
    [user@server-container $] ./notebooks/101_getting-started.py
    ```


## Misc server management commands

Some possible management commands after initial deployment include:

* check all containers running on the server side

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status vpnserver mqtt restful researcher
    ```

* check the VPN peers known from the VPN server

    ```bash
    [user@server $] ( cd ${FEDBIOMED_DIR}/envs/vpn/docker ; docker compose exec vpnserver bash -ci "python ./vpn/bin/configure_peer.py list" )
    type        id           prefix         peers
    ----------  -----------  -------------  ------------------------------------------------
    management  mqtt         10.220.0.2/32  ['1exampleofdummykey12345abcdef6789ghijklmnop=']
    management  restful      10.220.0.3/32  ['1exampleofdummykeyA79s0VsN5SFahT2fqxyooQAjQ=']
    researcher  researcher1  10.222.0.2/32  ['1exampleofdummykeyVo+lj/ZfT/wYv+I9ddWYzohC0=']
    node        NODETAG      10.221.0.2/32  ['1exampleofdummykey/Z1SKEzjsMkSe1qztF0uXglnA=']
    ```

* restart all containers running on the server side

    ```bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn stop vpnserver mqtt restful researcher
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn start vpnserver mqtt restful researcher
    ```

    VPN configurations and container files are kept unchanged when restarting containers.

* clean running containers, container files and temporary files on the server side. Requires to stop containers before.

    ``` bash
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn stop vpnserver mqtt restful researcher
    [user@server $] source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean
    ```

    Warning: all VPN configurations, researcher configuration files,experiment files and results, etc. are deleted when cleaning.

    To clean also the container images:

    ```
    [user@server $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image
    ```


## Misc node management commands

Some possible management commands after initial deployment include:

* check all containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn status node gui
    ```

* restart all containers running on the node side

    ```bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn stop node gui
    [user@node $] ( cd ${FEDBIOMED_DIR}/envs/vpn/docker ; docker compose up -d node gui )
    ```

    VPN configurations and container files are kept unchanged when restarting containers.

* clean running containers, container files and temporary files on the node side. Requires to stop containers before.

    ``` bash
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn stop node gui
    [user@node $] source ${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean
    ```

    Warning: all VPN configurations, node configuration files, node dataset sharing, etc. are deleted when cleaning.

    To clean also the container images:

    ```
    [user@node $] ${FEDBIOMED_DIR}/scripts/fedbiomed_vpn clean image
    ```

## Annex

### Proxy

On a site where access to an Internet web site requires using a proxy, configure [web proxy for docker client](https://docs.docker.com/network/proxy#configure-the-docker-client) in `~/.docker/config.json`.

Prefix used by Fed-BioMed's communication inside the VPN (`10.220.0.0/14`) shall not be proxied. So your proxy configuration may look like (replace `mysiteproxy.domain` with your site proxy):

```json
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://mysiteproxy.domain:3128",
     "httpsProxy": "http://mysiteproxy.domain:3128",
     "noProxy": "10.220.0.0/14"
   }
 }
}
```